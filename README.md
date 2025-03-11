# Mobile Steerer
## Introduction
This project introduces Mobile Steerer, a lightweight model for crowd counting. We used MobileNetV3 as the backbone and selected specific layers to fit the stage channel sizes efficiently. By optimizing the model architecture, we reduced the number of parameters to 3.9M. After training for 200 epochs, the model achieved an MAE of 85.73 and an RMSE of 152.78.

# Getting started 

- **Install dependencies.**

```bash
conda create -n STEERER python=3.9 -y
conda activate STEERER
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
cd ${STEERER}
pip install -r requirements.txt
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

# Reference

```
@article{haniccvsteerer,
  title={STEERER: Resolving Scale Variations for Counting and Localization via Selective Inheritance Learning},
  author={Han, Tao, Bai Lei, Liu Lingbo, and Ouyang  Wanli},
  booktitle={ICCV},
  year={2023}
}
```
