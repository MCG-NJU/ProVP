# [IJCV2024] Progressive Visual Prompt Learning with Contrastive Feature Re-formation
1. We propose a novel structure called Progressive Visual Prompt (ProVP). This design aims to strengthen the interaction among prompts from adjacent layers, thereby enabling more effective propagation to deeper layers.
2. We further introduce a contrastive feature reformation technique to address the generalization deterioration problem in training learnable prompts，whcih prevents significant deviations of prompted visual features from the fixed CLIP visual feature distribution.
3. This combined method, ProVP-Ref, is evaluated across 11 image datasets and achieves state-of-the-art results on 7/11 datasets in both few-shot learning and base-to-new generalization settings.
 <!--  To  the best of our knowledge, this is the first study to showcase the exceptional performance of visual prompts in V-L models compared to previous text prompting methods in this area.-->

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
