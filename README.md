# STEERER for Object Counting Baseline (ICCV 2023)
## Introduction
This is the official PyTorch implementation of paper: [**STEERER: Resolving Scale Variations for Counting and Localization via Selective Inheritance Learning**](https://arxiv.org/abs/2308.10468), which effectively addressed the issue of scale variations for object counting and localizaioion, demonstrating the state-of-arts counting and localizaiton performance for different catagories, such as crowd,vehicle, crops and trees ![framework](./figures/framework.png)


# Getting started 

## preparation 

- **Clone this repo** in the directory (```root/```):


```bash
cd $root
git clone https://github.com/taohan10200/STEERER.git
```
- **Install dependencies.** We use python 3.9 and pytorch >= 1.12.0 : http://pytorch.org.

```bash
conda create -n STEERER python=3.9 -y
conda activate STEERER
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
cd ${STEERER}
pip install -r requirements.txt
```

- <span style="color:red">**!!! Processed datasets and Pretrained-weights** </span> are available at the [OneDrive](https://pjlab-my.sharepoint.cn/:f:/g/personal/hantao_dispatch_pjlab_org_cn/EpdTPZDeIhxCpR5gr46iXyYBvEC1xo8qX96FvK8geMJs6Q?e=SGfrSS) net disk,  and you can selectively dolownd the dataset that you want to train or inference. Before starting your training and testing, you should organiza your project as the following directory tree. 

````bash

  $root/
  ├── ProcessedData
  │   ├── SHHB
  │   ├── SHHA
  │   ├── NWPU
  │   ├── QNRF
  │   │   ├── images     # the input images
  │   │   ├── jsons      # the annotated labels
  │   │   ├── train.txt   # the image name of train set 
  │   │   ├── test.txt    # the image name of test set
  │   │   ├── test_gt_loc.txt  # the localization labels for evaluation
  │   │   └──train_gt_loc.txt  # the localization labels for train set (not used)
  │   ├── JHU
  │   ├── MTC
  │   ├── JHU
  │   ├── JHUTRANCOS_v3
  │   └── TREE
  ├── PretrainedModels
  └── STEERER

````

## Training & Evaluation
we provide simplify script to run baseline model with A100 GPU in STEERER_train.ipynb file.

```bash
# Run this cell to train STEERER model in STEERER_train.ipynb file
sh ! python tools/train_cc.py --cfg=configs/QNRF_final.py --launcher="pytorch"

# Run this cell to test STEERER model in STEERER_train.ipynb file
! python tools/test_loc.py --cfg=configs/QNRF_final.py --checkpoint="exp/QNRF/MobileBackbone_mobile_large/QNRF_final_2024-12-17-15-14/Ep_427_mae_84.36152942451888_mse_150.99217291674904.pth" --launcher="pytorch"
```

## Reproduce Counting and Localization Performance
exp folder is deleted in this repository due to the storage limitation. You can check the checkpoint and output visualization image in Google Drive link.
link: https://drive.google.com/drive/folders/1-YQBecxXN-8lb152O4y3TbGgeeDZPexe?usp=sharing

|     Dataset     |     Method     |  MAE/MSE  | Dataset | Weight |
|------------|-------- |-------|-------|------|
| UCF-QNRF  |  STEERER   | 82.89/137.66 | [Dataset](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/Ef9E9oVtjyBEld_RYpPtqFUBfTBSy6ZgT0rqUhOMgC-X9A?e=WNn9aM)|Ep_471_mae_81.09296779289932_mse_134.13431722945182.pth|
| UCF-QNRF  |  STEERER (MobileNetV3 large)   | 85.73/152.78 | [Dataset](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/Ef9E9oVtjyBEld_RYpPtqFUBfTBSy6ZgT0rqUhOMgC-X9A?e=WNn9aM)|Ep_427_mae_84.36152942451888_mse_150.99217291674904.pth||
<!-- # References
1. Acquisition of Localization Confidence for Accurate Object Detection, ECCV, 2018.
2. Very Deep Convolutional Networks for Large-scale Image Recognition, arXiv, 2014.
3. Feature Pyramid Networks for Object Detection, CVPR, 2017.  -->

# Citation

```
@article{haniccvsteerer,
  title={STEERER: Resolving Scale Variations for Counting and Localization via Selective Inheritance Learning},
  author={Han, Tao, Bai Lei, Liu Lingbo, and Ouyang  Wanli},
  booktitle={ICCV},
  year={2023}
}
```

# Acknowledgement
The released PyTorch training script borrows some codes from the [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation) and [MMCV](https://github.com/open-mmlab/mmcv) repositories.
