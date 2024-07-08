import os.path as osp
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch.cuda.amp import GradScaler, autocast

from operator import mul
from functools import reduce

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_layers, num_tokens, hidden_size, p_drop=0, dtype=float):
        super().__init__()
        self.num_tokens = num_tokens
        self.p_drop = p_drop
        self.p_drops = list()
        self.dtype = dtype
        for _ in range(num_layers):
            self.p_drops.append(nn.Dropout(self.p_drop))
        # init prompt weights
        patch_size = 14
        val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size), 1) + hidden_size))  # noqa
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            num_layers, num_tokens, hidden_size))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

    def forward(self):
        prompts = torch.zeros_like(self.prompt_embeddings)
        for i in range(len(self.p_drops)):
            prompts[i] = self.p_drops[i](self.prompt_embeddings[i])
        return prompts
    def train(self, mode=True):
        super().train(mode)
        for drop in self.p_drops:
            if mode:
                drop.train()
            else:
                drop.eval()

CUSTOM_TEMPLATES = {
    # "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordPets": "a type of pet, a photo of a {}.",
    # "OxfordFlowers": "a photo of a {}, a type of flower.",
    "OxfordFlowers": "a type of flower, a photo of a {}.",
    "FGVCAircraft": "a type of aircraft, a photo of a {}.",
    "DescribableTextures": "a texture of {}.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    # "Food101": "a photo of {}, a type of food.",
    "Food101": "a type of food, a photo of {}.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class CLIP(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1,
                                                               keepdim=True)
        self.dtype = clip_model.dtype
        self.text_features = text_features
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        text_features = self.text_features
        text_features = text_features.to(image_features.device)
        logits = logit_scale * image_features @ text_features.t()
        return [logits, image_features.type(self.dtype)]


class mProVP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        text_prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Text Prompts: {text_prompts}")
        text_prompts = torch.cat([clip.tokenize(p) for p in text_prompts])

        self.text_prompts = text_prompts
        # text_features = clip_model.encode_text(text_prompts)
        # self.text_features = nn.Parameter(text_features / text_features.norm(dim=-1, keepdim=True))
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        # cfg.TRAINER.COOP.N_CTX
        print("Prompt Length:", cfg.TRAINER.COOP.N_CTX)
        self.prompt_learner = PromptLearner(12, cfg.TRAINER.COOP.N_CTX, 768, 0.0, clip_model.dtype)

    def get_text_prompts(self, device):
        text_prompts = self.text_prompts.to(device)
        text_features = self.clip_model.encode_text(text_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features

    def forward(self, image):
        prompt_embeddings = self.prompt_learner()
        image_features = self.clip_model.visual(image.type(self.dtype), prompt_embeddings.type(self.dtype))
        text_features = self.text_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return [logits, image_features.type(self.dtype)]


class ProGradLoss(_Loss):
    def __init__(self, T):
        super(ProGradLoss, self).__init__()
        self.T = T

    def forward(self, stu_logits, tea_logits, label):
        xe_loss = F.cross_entropy(stu_logits, label)

        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T,
                                            -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()

        return xe_loss, kl_loss


class ReformatLoss(_Loss):
    def __init__(self, T):
        super(ReformatLoss, self).__init__()
        self.T = T

    def forward(self, p_output, p_feat, clip_feat, sym_label, label):
        xe_loss = F.cross_entropy(p_output, label)
        ref_loss = F.cross_entropy(p_feat @ clip_feat.type_as(p_feat).t(), sym_label)
        return xe_loss, ref_loss


@TRAINER_REGISTRY.register()
class ProVP(TrainerX):
    """Progressive Visual Prompt with Contrastive Feature Re-formation
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg

        self.device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
        if cfg.TRAINER.COOP.N_CTX == 16:
            self.device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
            if self.cfg.DATASET.NAME == "ImageNet":
                self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
            if self.cfg.DATASET.NAME == "UCF101":
                self.device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
            if self.cfg.DATASET.NAME == "StanfordCars":
                self.device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
            if self.cfg.DATASET.NAME == "EuroSAT":
                self.device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
            if self.cfg.DATASET.NAME == "DescribableTextures":
                self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        if cfg.TRAINER.COOP.N_CTX == 50:
            self.device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
            if self.cfg.DATASET.NAME == "DescribableTextures":
                self.device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
            elif cfg.DATASET.NAME == "StanfordCars":
                self.device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
            elif self.cfg.DATASET.NAME == "UCF101":
                self.device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        print("Building zeroshot CLIP")
        self.zs_clip = CLIP(cfg, classnames)

        print("Building custom CLIP")
        self.model = mProVP(cfg, classnames, clip_model)

        print("Turning off gradients in ZS Clip model")
        for name, param in self.zs_clip.named_parameters():
            param.requires_grad_(False)

        print("Turning off gradients in CoOp model")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner,
                                    cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.get_text_prompts(self.device)
        self.zs_clip.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # self.register_model("prompt_learner", self.model.prompt_learner,
        #                           self.optim, self.sched)
        self.register_model("Prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # build criterion
        if cfg.OPTIM.USE_REF == True:
            self.use_ref = True
            self.criterion = ReformatLoss(T=cfg.LOSS.T)#ProGradLoss(T=cfg.LOSS.T)
        else:
            self.use_ref = False

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC

        if self.use_ref:
            if prec == "amp":
                with autocast():
                    output, p_feat, _ = self.model(image)
                    with torch.no_grad():
                        zs_clip_output, clip_feat = self.zs_clip(image)
                    sym_label = torch.arange(clip_feat.shape[0]).to(self.device)
                    loss = self.criterion(output, zs_clip_output, label)
                    # loss = self.criterion(output, p_feat, clip_feat.detach(), sym_label, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output, p_feat= self.model(image)
                with torch.no_grad():
                    zs_clip_output, clip_feat = self.zs_clip(image)
                sym_label = torch.arange(clip_feat.shape[0]).to(self.device)
                loss, ref_loss = self.criterion(output, p_feat, clip_feat, sym_label, label)
                Lam = 1.0
                if self.cfg.DATASET.NAME == "StanfordCars" or  self.cfg.DATASET.NAME == "ImageNet" or self.cfg.DATASET.NAME == "OxfordFlowers":
                    Lam = 1.0
                if self.cfg.TRAINER.COOP.N_CTX == 8 and self.cfg.DATASET.NAME != "ImageNet":
                    Lam = 1.0
                ref_loss = ref_loss * Lam
                self.model_backward_and_update(loss+ref_loss)
            loss_summary = {
                "xe_loss": loss.item(),
                "ref_loss": ref_loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }

        else:
            if prec == "amp":
                with autocast():
                    output, _ = self.model(image)
                    loss = F.cross_entropy(output, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output, _= self.model(image)
                loss = F.cross_entropy(output, label)
                self.model_backward_and_update(loss)

            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained model is given"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict)