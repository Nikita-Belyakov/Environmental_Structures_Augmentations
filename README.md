# **CSIA: Climate Structures Inpainting Augmentations for Multispectral Remote Sensing Imagery segmentation**

## **About:**

This repo is dedicated to novel custom augmentations for semantic segmentation task of environmental & climate structures, namely, clouds, it's shadows and snow for remote sensing multispectral imagery.

Augmentation requires input image sample and its mask. In these .py augmnetation cloud, its shadows and snow structures are augmneted for Landsat-8 multispectral imagery from SPARCS validation dataset: https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data

We attach a trained U-Net++ segmentation model weights (from SMP: https://smp.readthedocs.io/en/latest/models.html), able to do accurate segmentation of clouds, it's shadows and snow for Landsat multispectral imagery. Each model weights are stored in **models** folder, which is seperated to 2 subfolder: **inpainting_models** & **segmentation_models**, each containing model weights for each CSIA mode.

<center><img src="CSIA_aug_scheme2.png" width="800"></center>

## **ABSTRACT**
Today convolutional neural networks (CNNs) models outperform most of the computer vision tasks, providing state-of-the-art (SotA) results comparable with human performance. In remote sensing (RS) semantic segmentation problems CNNs also show great performance. However, for accurate segmentation, CNN models usually require a lot of high-quality training data, that is usually can be evaluated only with human labeling. Rare structures and its variability as well as different environmental and climate conditions strongly influence on the stability and robustness of CNNs performance. To improve the segmentation quality with lack of training data, it is considered to use various approaches including data augmentation techniques. This research is focused on the development and testing of climate-structures-based augmentations, that significantly improves the segmentation performance for the hard distinguished classes, namely, snow, clouds and its shadows. We show the practical application of the developed augmentations technique on the RS imagery from Landsat-8. We propose a new pipeline for performing CNN inpainting image augmentations with different climate structures on RS scenes, that helps to raise the variability of training data. We called the proposed technique climate structures inpainting augmentations (CSIA). It expands training samples with new realistic clouds, its shadows and snow structures on label-free backgrounds RS scenes. We test our approach on the Spatial Procedures for Automated Removal of Cloud and Shadow (SPARCS) dataset with Landsat-8 multispectral imagery (MSI) with the most usable U-Net CNN architecture and show that the proposed augmentation technique benefits for all the tested segmentation classes. CSIA leads to the meaningful improvement of U-Net model predictions from 0.65 to 0.68 IoU-score for the most difficult clouds shadows segmentation class.
**Keywords: remote sensing; convolutional neural network; semantic segmentation; custom augmentations; image restoration; cloud and its shadow inpainting; Landsat-8 imagery**

Examples of the improvement for climate structures segmentataion performance are illustrated below:

<center><img src=".pics/segmentation_examples/predict_BASE_vs_CSIA13" width="800"></center>

## **CITE**: https://doi.org/10.1016/j.asr.2025.01.049

```
@article{BELYAKOV2025,
title = {CSIA: Climate Structures Inpainting Augmentations for Multispectral Remote Sensing Imagery segmentation},
journal = {Advances in Space Research},
year = {2025},
issn = {0273-1177},
doi = {https://doi.org/10.1016/j.asr.2025.01.049},
url = {https://www.sciencedirect.com/science/article/pii/S0273117725000791},
author = {Nikita V. Belyakov and Svetlana Illarionova},
keywords = {remote sensing, convolutional neural network, semantic segmentation, custom augmentations, image restoration, cloud and its shadow inpainting, Landsat-8 imagery}
```

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
