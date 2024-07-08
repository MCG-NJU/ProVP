import os.path as osp

import torch
import math
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
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


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, deep_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x[:, :min(x.shape[1], tokenized_prompts.argmax(dim=-1).max()+1)]
        print(x.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, deep_prompts)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class TextPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.TEXT_NCTX
        n_layers = cfg.TRAINER.COOP.TEXT_LAYER
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        ## init prompt
        prompt_prefix = "a photo of a"
        prompt = clip.tokenize(prompt_prefix)

        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)

        if n_ctx <= 4:
            if n_ctx > 1:
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            else:
                ctx_vectors = embedding[0, 2 : 3, :]
        else:
            prompt_prefix = (" ".join(["X"] * (n_ctx - 4))) + " " + prompt_prefix
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            ctx_vectors[-n_ctx:] = embedding[0, 1 : 1 + n_ctx, :]

        if n_layers > 1:
            ctx_vectors_deep = torch.empty(n_layers-1, n_ctx, ctx_dim, dtype=dtype)
            for i in range(n_layers-1):
                nn.init.normal_(ctx_vectors_deep[i], std=0.02)
            ctx_vectors = torch.cat([ctx_vectors.unsqueeze(0), ctx_vectors_deep], dim=0)
        else:
            ctx_vectors = ctx_vectors.unsqueeze(0)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # temp = "{}."
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        classnames = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames]
        print("Prompts:", prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        ctx = self.ctx[0]
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat([prefix, ctx, suffix], dim=1)

        return prompts, self.ctx


class VisualPromptLearner(nn.Module):
    def __init__(self, num_layers, num_tokens, hidden_size):
        super().__init__()
        self.num_tokens = num_tokens
        # init prompt weights
        patch_size = 14
        val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size), 1) + hidden_size))  # noqa
        self.prompt_embeddings = nn.Parameter(torch.zeros(num_layers, num_tokens, hidden_size))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

    def forward(self):
        return self.prompt_embeddings

class PromptLearner(nn.Module):
    def __init__(self, clip_model, cfg, classnames):
        super().__init__()
        self.vp_learner = VisualPromptLearner(cfg.TRAINER.COOP.VISUAL_LAYER,
                                              cfg.TRAINER.COOP.VISUAL_NCTX,
                                              clip_model.visual.ln_pre.weight.shape[0])
        self.tp_learner = TextPromptLearner(cfg, classnames, clip_model)

    def forward(self):
        vprompts = self.vp_learner()
        tprompts, tprompts_deep = self.tp_learner()
        return [vprompts, tprompts, tprompts_deep]
    
CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, which is a type of an aircraft.",
    "DescribableTextures": "a photo of a {}, a type of a texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a type of food, a photo of {}.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}, a type of rendition.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}, a type of sketch.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}, which is natural adversarial.",
    "ImageNetR": "a photo of a {}, a type of rendition.",
}

class CLIP(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        temp = "a photo of a {}."
        # temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.dtype = clip_model.dtype
        self.text_features = text_features
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

        print(f"Visual Prompt Length: {cfg.TRAINER.COOP.VISUAL_NCTX}, Layers: {cfg.TRAINER.COOP.VISUAL_LAYER}", )
        print(f"Text Prompt Length: {cfg.TRAINER.COOP.TEXT_NCTX}, Layers: {cfg.TRAINER.COOP.TEXT_LAYER}", )

        self.prompt_learner = PromptLearner(clip_model, cfg, classnames)

        self.text_encoder = TextEncoder(clip_model)
        self.tokenized_prompts = self.prompt_learner.tp_learner.tokenized_prompts

        self.image_features_unnorm = None
        self.text_features_unnorm = None
        self.text_features = None

    def forward(self, image, eval=False):
        vprompts, tprompts, tprompts_deep = self.prompt_learner()

        image_features = self.clip_model.visual(image.type(self.dtype), vprompts.type(self.dtype))
        self.image_features_unnorm = image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if eval == True:
            if self.text_features == None:
                text_features = self.text_encoder(tprompts, self.tokenized_prompts, tprompts_deep)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.text_features = text_features
            else:
                text_features = self.text_features
        else:
            self.text_features = None
            text_features = self.text_encoder(tprompts, self.tokenized_prompts, tprompts_deep)
            self.text_features_unnorm = text_features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class ProMP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        if self.cfg.DATASET.NAME == "ImageNet":
                self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        if self.cfg.DATASET.NAME == "SUN397":
            self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        if self.cfg.DATASET.NAME == "StanfordCars":
            self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        if self.cfg.DATASET.NAME == "EuroSAT":
            self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        if self.cfg.DATASET.NAME == "DescribableTextures":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif self.cfg.DATASET.NAME == "Food101":
                self.device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        elif self.cfg.DATASET.NAME == "Caltech101":
                self.device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.zs_clip = CLIP(cfg, classnames)

        print("Turning off gradients in CoOp model")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        for name, param in self.zs_clip.named_parameters():
            param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner,cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.zs_clip.to(self.device)
        # t_optim = build_optimizer(self.model.prompt_learner.tp_learner, cfg.OPTIM, lr=cfg.OPTIM.LR_TEXT)
        # v_optim = build_optimizer(self.model.prompt_learner.vp_learner, cfg.OPTIM, lr=cfg.OPTIM.LR_VISUAL)
        if cfg.DATASET.NAME in ['EuroSAT']:
            t_optim = build_optimizer(self.model.prompt_learner.tp_learner, cfg.OPTIM, lr=cfg.OPTIM.LR_TEXT * 0.0) 
            v_optim = build_optimizer(self.model.prompt_learner.vp_learner, cfg.OPTIM, lr=cfg.OPTIM.LR_VISUAL)
        else:
            t_optim = build_optimizer(self.model.prompt_learner.tp_learner, cfg.OPTIM, lr=cfg.OPTIM.LR_TEXT)
            v_optim = build_optimizer(self.model.prompt_learner.vp_learner, cfg.OPTIM, lr=cfg.OPTIM.LR_VISUAL)

        v_sched = build_lr_scheduler(v_optim, cfg.OPTIM)
        self.register_model("visual_prompt", self.model.prompt_learner.vp_learner, v_optim, v_sched)

        t_sched = build_lr_scheduler(t_optim, cfg.OPTIM)
        self.register_model("text_prompt", self.model.prompt_learner.tp_learner, t_optim, t_sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        self.use_ref = cfg.OPTIM.USE_REF

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        output = self.model(image)
        if self.use_ref:
            # visual ref loss
            clip_visual_feat = self.zs_clip(image)
            image_features = self.model.image_features_unnorm / self.model.image_features_unnorm.norm(dim=-1, keepdim=True)
            visual_sym_label = torch.arange(label.shape[0]).to(self.device)

            visual_ref_loss = F.cross_entropy(image_features @ clip_visual_feat.type_as(image_features).t(), visual_sym_label)

            # text ref loss
            clip_text_feat = self.zs_clip.text_features
            text_features = self.model.text_features_unnorm / self.model.text_features_unnorm.norm(dim=-1, keepdim=True)
            text_sym_label = torch.arange(text_features.shape[0]).to(self.device)

            text_ref_loss = F.cross_entropy(text_features @ clip_text_feat.type_as(text_features).t(), text_sym_label)

            # ce loss
            ce_loss = F.cross_entropy(output, label)
            if self.cfg.DATASET.NAME == "ImageNet":
                Lamb = 0.1
            else:
                Lamb = 1.0
            Lamb_text = Lamb
            DATANAME = self.cfg.DATASET.NAME
            if DATANAME == "EuroSAT":
                Lamb_text = Lamb_text * 0.8
            elif DATANAME == "FGVCAircraft":
                Lamb_text = Lamb_text * 0.4
            loss = Lamb * visual_ref_loss + Lamb_text * text_ref_loss + ce_loss
            # loss = Lamb * visual_ref_loss + ce_loss
            loss_summary = {
                "ce_loss": ce_loss.item(),
                "visual_ref_loss": visual_ref_loss.item() * Lamb,
                "text_ref_loss": text_ref_loss.item() * Lamb_text,
                "acc": compute_accuracy(output, label)[0].item(),
            }

        else:
            loss = F.cross_entropy(output, label)
            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }
        self.model_backward_and_update(loss)

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
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        else:
            model_file = "model-best.pth.tar"

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                model_file_last = "model.pth.tar-" + str(self.epoch+1)
                model_path_last = osp.join(directory, name, model_file_last)
                if osp.exists(model_path_last):
                    model_path = model_path_last
                else:
                    raise FileNotFoundError('Model not found at "{}" and "{}"'.format(model_path, model_path_last))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
