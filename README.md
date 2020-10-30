# BEDs
Tensorflow based implementation of BEDs as described in BEDs: Bagging Ensemble Deep Segmentation for Nucleus Segmentation with Testing Stage Stain Augmentation.

## Setup
This repo tested in following environment:
```bash
python==3.5.2
tensorflow-gpu==1.12.0
opencv-python==3.4.0.14
```
To setup the python environment for this project, run:
```bash
pip install -r requirements.txt
```
Follow the procedure below to setup and download necessary data to reproduce the results in the paper.
1. Clone this repository.
2. Create directory for datasets.
```bash
mkdir datasets
cd datasets
mkdir Train Val Test
```
3. Download training dataset from [manually labeled dataset](https://app.box.com/s/fz425ixs15kf56ghbnpxng1es6m7v2oh): Dataset of segmented nuclei in hematoxylin and eosin stained histopathology images of ten cancer types. Extract to `datasets/Train`. Put `*_crop.png` to `datasets/Train/Images` and `*_labeled_mask_corrected.png` to `datasets/Train/Annotations`.
3. Download validation dataset from [MoNuSeg Challenge 2018 (Training)](https://drive.google.com/file/d/1JZN9Jq9km0rZNiYNEukE_8f0CsSK3Pe4/view). Extract to `datasets/Val`. Rename the directory `Tissue Images` to `Images`
4. Download testing dataset from [MoNuSeg Challenge 2018 (Testing)](https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view). Extract to `datasets/Test`. Put `.tif` files to `datasets/Test/Images` and `.xml` files to `datasets/Test/Annotations`.
5. (Optional) Download our stain augmented testing images [here](https://drive.google.com/file/d/1VvFbE0kKD85rLZjK0T1L4Rh0NR4Xfbt7/view?usp=sharing). Extract to `datasets/Test`
6. (Optional) Download our pre-trained models [here](https://drive.google.com/file/d/13mx5xXMtHRQ7iUJuPJCtnL9RaV_2vW2y/view?usp=sharing). Extract to `BEDs`.

## Data Processing

After following the steps in Setup, run following script to:
```bash
cd BEDs/
```
1. Generate benchmark and random split training data for U-net training:
```bash
python data_process/prepare_datasets.py --annot-dir datasets/Train/Annotations/ --output-dir datasets/Train/deep_forest/ --stage train --subset-num 33 datasets/Train/Images/
```
2. Generate validation data:
```bash
python data_process/prepare_datasets.py --annot-dir datasets/Val/Annotations/ --output-dir datasets/Val/Val/ --stage val datasets/Val/Images/
```
3. Generate testing data for each type of stain augmentation (0 is the original stain, 1-6 is the augmented stain):
```bash
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotation/ --output-dir datasets/Test/Test_pairs/0/ --stage test datasets/Test/Images_stainNormed/0/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotation/ --output-dir datasets/Test/Test_pairs/1/ --stage test datasets/Test/Images_stainNormed/1/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotation/ --output-dir datasets/Test/Test_pairs/2/ --stage test datasets/Test/Images_stainNormed/2/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotation/ --output-dir datasets/Test/Test_pairs/3/ --stage test datasets/Test/Images_stainNormed/3/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotation/ --output-dir datasets/Test/Test_pairs/4/ --stage test datasets/Test/Images_stainNormed/4/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotation/ --output-dir datasets/Test/Test_pairs/5/ --stage test datasets/Test/Images_stainNormed/5/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotation/ --output-dir datasets/Test/Test_pairs/6/ --stage test datasets/Test/Images_stainNormed/6/
```

## Inference with pre-trained models

## Experiments and Evaluation

## Train U-net by yourself

