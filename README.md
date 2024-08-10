# [IJCV2024] Progressive Visual Prompt Learning with Contrastive Feature Re-formation

We propose a novel structure called Progressive Visual Prompt (ProVP). This design aims to strengthen the interaction among prompts from adjacent layers, thereby enabling more effective propagation of image embeddings to deeper layers in a manner akin to an instance-specific manner. Additionally, to address the common issue of generalization deterioration in the training period of learnable prompts, we further introduce a contrastive feature reformation technique for visual prompt learning. This method prevents significant deviations of prompted visual features from the fixed CLIP visual feature distribution, ensuring better generalization capability. Combining the ProVP and the contrastive feature Reformation technique, our proposed method, ProVPRef, significantly stabilizes the training process and enhances both the adaptation and generalization capabilities of visual prompt learning in V-L models posttraining. To demonstrate the efficacy of our approach, we evaluate ProVP-Ref across 11 image datasets, achieving the state-of-the-art results on 7 of these datasets in both few-shot learning and base-to-new generalization settings. To the best of our knowledge, this is the first study to showcase the exceptional performance of visual prompts in V-L models compared to previous text prompting methods in this area.

![image](/model.jpg)

[[paper link]](https://doi.org/10.48550/arXiv.2304.08386)

The codes are organized into two folders:

1. [Dassl](Dassl/) is the modified toolbox of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) which supports the implementation of ProVP
2. [ProVP](ProVP/). 

## Citation

If you find our paper or this project helps your research, please kindly consider citing our paper in your publication.

```
@misc{xu2024progressivevisualpromptlearning,
      title={Progressive Visual Prompt Learning with Contrastive Feature Re-formation}, 
      author={Chen Xu and Yuhan Zhu and Haocheng Shen and Fengyuan Shi and Boheng Chen and Yixuan Liao and Xiaoxin Chen and Limin Wang},
      year={2024},
      eprint={2304.08386},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.08386}, 
}

```

## Acknowledgement
Our codes are built on top of [ProGrad](https://github.com/BeierZhu/Prompt-align/) and [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch).
