# **CSIA: Climate Structures Inpainting Augmentations for Multispectral Remote Sensing Imagery segmentation**

<img src="./RES_4_KM/l2_MANet_predict.png" width="800"/>

## **About:**

This repo is dedicated to novel custom augmentations for semantic segmentation task of environmental & climate structures, namely, clouds, it's shadows and snow for remote sensing multispectral imagery.

Augmentation requires input image sample and its mask. In these .py augmnetation cloud, its shadows and snow structures are augmneted for Landsat-8 multispectral imagery from SPARCS validation dataset: https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data

We attach a trained U-Net++ segmentation model weights (from SMP: https://smp.readthedocs.io/en/latest/models.html), able to do accurate segmentation of clouds, it's shadows and snow for Landsat multispectral imagery. Each model weights are stored in **models** folder, which is seperated to 2 subfolder: **inpainting_models** & **segmentation_models**, each containing model weights for each CSIA mode.

## Setup python version
All .py augmentation files from **Augmentations** folder have been run with `python 3.9.7` on Windows 10 OS with NVIDIA CUDA supported (Adapt all needed packages versions accroding your Python version)

### All required packages are provided in requirements.txt
- It's recomended to use `Pytorch` version with CUDA support! To install pytorch with cuda run appropriate command in your console from here:
  - https://pytorch.org/get-started/locally/
  - (We used this version of cudann: `pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`)
- Just run in your .ipynb this cell:
```
 !pip install -r requirements.txt
```
## Acknowledgments:

Also thanks a lot to Svetlana Illarionova (https://github.com/LanaLana) for a huge assist for creating this project!
