# MNIST_DDRGB_ObjectDetection


## Table of Contents
- [Introduction](#introduction)
- [Dataset](#Dataset)
- [Results](#Results)
- [MNISTDD-RGB-Dataset-Illustration](#MNISTDD-RGB-Dataset-Illustration)
- [References](#References)
- [Contact](#contact)

## Introduction
This initiative centers on the detection of objects within the MNIST Double Digits RGB dataset. It aims to identify and categorize two numerical digits in RGB images while also determining their bounding boxes. The project harnesses DarkNet53 for efficient object detection and UNet for precise Image Segmentation, resulting in impressive scores in classification, Intersection over Union (IOU), and image segmentation.

## Dataset
The dataset is divided into three parts: training, validation, and testing, with 55,000, 5,000, and 10,000 samples in each, respectively. It includes:

Inputs: Flattened images as numpy arrays, each of size 64x64x3, resulting in arrays of shape (number of samples, 12288).
Labels: 2D vectors for each image, containing two digits ranging from 0 to 9, representing the digits present in the image.
Segmentation Masks: Each a 64×64 image where pixel values range from 0 to 10, with 10 indicating the background.
The expected outputs are:

Classifications: Numpy arrays specifying the classes of the digits in the images, shaped (number of samples, 2).
Predicted Bounding Boxes: Numpy arrays of the bounding boxes for each image, formatted as (number of samples, 2, 4).
  
## Results
- Classification Accuracy: 98.050 %
- Detection IOU: 92.538 %
- Segmentation Accuracy: 99.669 %
- Test time: 74.939 seconds
- Test speed: 66.721 images / second
- Classification Score: 100.000
- IOU Score: 80.494
- Segmentation Score: 100.000
- Overall Score: 95.124

## MNISTDD-RGB-Dataset-Illustration

Here is an example image from the MNISTDD-RGB dataset to give you an idea of what the dataset looks like:

![MNISTDD-RGB Sample Image](dataset.png)

## References
-  UNet Model : https://github.com/milesial/Pytorch-UNet/tree/master

## Contact
- If you need weights file please mail me : rishabhmehra2710@gmail.com

