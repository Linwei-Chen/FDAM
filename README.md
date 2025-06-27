# FDAM: Frequency-Dynamic Attention Modulation

**[ICCV 2025]** Official implementation of *Frequency-Dynamic Attention Modulation for Dense Prediction*.
FDAM revitalizes Vision Transformers by tackling frequency vanishing. It dynamically modulates the frequency response of attention layers, enabling the model to preserve critical details and textures for superior dense prediction performance.

The code is being organized, and will be released soon.

## 🚀 Key Features

- **Attention Inversion (AttInv):** Inspired by circuit theory, AttInv inverts the inherent low-pass filter of the attention mechanism to generate a complementary high-pass filter, enabling a full-spectrum representation.
- **Frequency Dynamic Scaling (FreqScale):** Adaptively re-weights and amplifies different frequency bands in feature maps, providing fine-grained control to enhance crucial details like edges and textures.
- **Prevents Representation Collapse:** Effectively mitigates frequency vanishing and rank collapse in deep ViTs, leading to more diverse and discriminative features.
- **Plug-and-Play & Efficient:** Seamlessly integrates into existing ViT architectures (like DeiT, SegFormer, MaskDINO) with minimal computational overhead.

![image-20250214164031767](README.assets/fdam.png)

## 📈 Performance Highlights

### Semantic Segmentation (ADE20K val set)

FDAM boosts performance on various backbones, including CNN-based, ViT-based, and even recent Mamba-based models.

| Backbone     | Base mIoU (SS) | + FDAM mIoU (SS) | Improvement |
| ------------ | -------------- | ---------------- | ----------- |
| SegFormer-B0 | 37.4           | **39.8**         | **+2.4**    |
| DeiT-S       | 42.9           | **44.3**         | **+1.4**    |

### Object Detection & Instance Segmentation (COCO val2017)

Integrated into the state-of-the-art Mask DINO framework, FDAM achieves notable gains with minimal overhead. * indicates reproduced results.

| Task                  | Method           | Metric | Baseline | + FDAM   | Improvement |
| --------------------- | ---------------- | ------ | -------- | -------- | ----------- |
| Object Detection      | Mask DINO (R-50) | APbox  | 45.5*    | **47.1** | **+1.6**    |
| Instance Segmentation | Mask DINO (R-50) | APmask | 41.2*    | **42.6** | **+1.4**    |
| Panoptic Segmentation | Mask DINO (R-50) | PQ     | 48.7*    | **49.6** | **+0.9**    |

### Remote Sensing Object Detection (DOTA-v1.0)

FDAM achieves **state-of-the-art** results in single-scale settings, showcasing its effectiveness in specialized domains.

| Backbone | Base mAP | + FDAM mAP | Improvement |
| -------- | -------- | ---------- | ----------- |
| LSKNet-S | 77.49    | **78.61**  | **+1.12**   |

## 🛠 Installation

Our implementation is primarily based on [mmdetection](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Fopen-mmlab%2Fmmdetection) and [mmsegmentation](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Fopen-mmlab%2Fmmsegmentation). Please follow their official guides for installation.

1. Install PyTorch and torchvision.
2. Install mmcv.
3. Install mmdetection and mmsegmentation.
4. Clone this repository.

## 📖 Citation

If you find this work useful for your research, please consider citing our paper:

Generated code

```
@InProceedings{chenlinwei2025ICCV,
  title={Frequency-Dynamic Attention Modulation for Dense Prediction},
  author={Chen, Linwei and Gu, Lin and Fu, Ying},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## Acknowledgment

This project is built upon the great work from several open-source libraries, including:

- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [MaskDINO](https://github.com/IDEA-Research/MaskDINO)

We thank their authors for making their code publicly available.

## Contact

If you have any questions or suggestions, please feel free to open an issue or contact us at [charleschen2013@163.com]. We welcome any feedback and discussion.