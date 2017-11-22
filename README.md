# MARTA-GAN
This is the code for [MARTA GANs: Unsupervised Representation Learning for Remote Sensing Image Classification](https://arxiv.org/abs/1612.08879). An multiple-layer feature-matching generative adversarial networks (MARTA GANs) to learn a representation using only unlabeled data.  



## Prepare data

Download and unzip dataset from [BaiDuYun](https://pan.baidu.com/s/1i5zQNdj) or [Google Drive](https://drive.google.com/open?id=0B1Evui8Soh85ZXM3cDNvbGdOamc).

## Dependencies

NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN mode are also available but significantly slower)

- tensorflow
- tensorlayer
- sklearn

## Usage

Training GAN
```
python train_marta_gan.py
```

Extract features
```
python extract_feature.py
```

Training SVM

```
python train_svm.py
```

## Citation
If you find this code useful for your research, please cite:
```
@article{lin2017marta,
  title={MARTA GANs: Unsupervised Representation Learning for Remote Sensing Image Classification},
  author={Lin, Daoyu and Fu, Kun and Wang, Yang and Xu, Guangluan and Sun, Xian},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2017},
  publisher={IEEE}
}
```
