## Reproducing DeiT-III-B + FDAM on ADE20K

This guide provides detailed instructions to reproduce the semantic segmentation results of the **DeiT-III-Base** model enhanced with **FDAM**, as reported in our paper. The model achieves **52.6 mIoU** on the ADE20K validation set.

### 1. Environment Setup

Before proceeding, please ensure you have completed the full installation as described in the [main `README.md`](../README.md) of this repository. This includes installing PyTorch, `mmcv-full`, and `mmsegmentation`.

```
pip install -r requirements.txt
```

### 2. Dataset Preparation

This experiment requires the **ADE20K** dataset.

1.  **Download:** Download the dataset from the [official MIT server](https://ade20k.csail.mit.edu/).
2.  **Directory Structure:** Organize the dataset files into the following structure under the project's root `data/` folder. The MMSegmentation framework expects this layout by default.

    ```text
    FDAM/
    ├── data/
    │   └── ade/
    │       └── ADEChallengeData2016/
    │           ├── annotations/
    │           │   ├── training/
    │           │   └── validation/
    │           ├── images/
    │           │   ├── training/
    │           │   └── validation/
    │           └── objectInfo.txt
    ├── ... (other project files)
    ```
3. For more details on dataset preparation, you can refer to the official [MMSegmentation guide](https://github.com/open-mmlab/mmsegmentation/blob/0.x/docs/en/user_guides/2_dataset_prepare.html).

### 3. Training from Scratch

The original experiment was performed on a server with **4x NVIDIA 3090 GPUs**. The provided configuration is optimized for this setup.

To start the training process, run the `dist_train.sh` script from the **root directory** of the repository:

```bash
# Set the number of GPUs
GPUS=4

# Run the distributed training script
./tools/dist_train.sh \
    FDAM_mmseg/configs/vit/upernet_deit3-b16_512x512_160k_ade20k_freq.py \
    $GPUS \
    --work-dir work_dirs/deit3-b-fdam-ade20k
```

-   **Download:** Download the pretrain Deit-III-Base from the [link](https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth), and modify the config.
-   The training logs and model checkpoints will be saved in the `work_dirs/deit3-b-fdam-ade20k/` directory.
-   If you use a different number of GPUs, you may need to adjust the learning rate and batch size in the config file `upernet_deit3-b16_512x512_160k_ade20k_freq.py` to achieve optimal results.

### 4. Evaluation

You can evaluate your trained model or the provided pre-trained model.

#### Evaluating Your Trained Model

Once training is complete, the final checkpoint will be available in your working directory. Run the following command to evaluate its performance on the ADE20K validation set.

```bash
# Set the number of GPUs for evaluation
GPUS=4

# IMPORTANT: Set this to the actual path of your best checkpoint
CHECKPOINT_PATH=work_dirs/deit3-b-fdam-ade20k/iter_160000.pth

# Run the distributed evaluation script
./tools/dist_test.sh \
    FDAM_mmseg/configs/vit/upernet_deit3-b16_512x512_160k_ade20k_freq.py \
    $CHECKPOINT_PATH \
    $GPUS \
    --eval mIoU
```

#### Evaluating the Pre-trained Model

We provide the pre-trained weights to directly reproduce our results.

1.  **Download the model:** [**DeiT-III-B + FDAM weights**](https://pan.baidu.com/s/1bylU0PojPlbsE1-ERbB05w?pwd=ICCV).
2.  Run the evaluation script with the path to the downloaded weights:

```bash
# Set the number of GPUs
GPUS=4

# Path to the pre-trained model you downloaded
PRETRAINED_MODEL_PATH=/path/to/your/deit3-b-fdam-model.pth

# Run evaluation
./tools/dist_test.sh \
    FDAM_mmseg/configs/vit/upernet_deit3-b16_512x512_160k_ade20k_freq.py \
    $PRETRAINED_MODEL_PATH \
    $GPUS \
    --eval mIoU
```
