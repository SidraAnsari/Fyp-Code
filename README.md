# Breast Cancer Image Analysis using Classification and Translation of H&E:

My research uses image classification and translation to diagnose breast cancer disease using hematoxylin and eosin images. Worldwide, breast cancer is the leading cause of death from cancer among women. This study investigates the use of deep learning models to histopathology image-based breast cancer diagnosis and therapy. These photos help identify cellular and tissue components by examining microscopic, color-coded breast cancer tissue slices. First, we used CNN for Image Classification, but this takes a lot of time. Therefore, we moved to Transfer Learning Model, which uses pre-trained models like SOTA model Vision Transformers utilizing BreakHis Dataset. The identification of cellular and tissue components requires the use of H&E pictures. Immunohistochemistry pictures, on the other hand, show the spatial characteristics of particular cell types. In Image-to-Image Translation studies, a suggested model called CycleGAN is utilized to generate/translate immunohistochemistry pictures from Hematoxylin and Eosin images for the treatment of breast cancer.


# Setup:Envs
-Windows<br>
-Python>=3.11<br>
-NVIDIA GPU 

# Framework Used:
-Pytorch

# Datasets Used:
For the Classification working in this study, we employed two benchmark datasets: the BreakHis (Breast Cancer Histopathological) dataset, which is made up of pictures of breast histopathology. The BCI (Breast Cancer Immunohistochemical) dataset, on the other hand, is specifically used for picture production and translation of paired H&E to IHC images.
## BreakHis
BreakHis is regarded as the most well-liked and therapeutically significant public histopathology dataset for breast cancer<br>. Using various amplification factors (40X, 100X, 200X, and 400X), 7,909 microscopic stained pictures of surgical biopsy (Sob)breast tumor tissue obtained from 82 individuals are included. It now has 5,429 cancerous and 2,480 benign samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG arrangement). 
- [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
- ![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/Breakhis%20dataset.jpg)

## BCI

BCI is a composite dataset that translates images of HER2 expression from H&E to immunohistochemical findings and HE-stained slices to images.4870 recorded picture pairings with various HER2 activity levels are present in this database.
- [BCI](https://bci.grand-challenge.org/dataset/)
 ![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/datasetpreview6.png)
# Methods:
## Image Classification :

The architecture of the pre-trained model is employed as a feature extractor in transfer learning. The model's first layers, which pick up on <br> fundamental visual characteristics like edges and textures, are kept. Because these layers are frozen, the known components are maintained during training without updating their weights.<br>
The breast image <br>dataset is used to adjust and retrain the pre-trained model's later layers, which capture more sophisticated and task-specific properties. This process is known as fine-tuning. Through adjustments to these layers' weights, the model adjusts to the unique features of the cancer picture classification job<br>.

![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/TL%20new.drawio.png)

# Models:
# Image Classification Models Used:
## Vision Transformers
<br>
The Vision Transformer (ViT) is the base of numerous Transformer blocks. The ViT divides the input histopathological image into fixed-size patches that may be thought of as tokens using the BreakHis dataset. Next, we apply a Positional Encoding layer, which ensures that our picture patches match the input order by linearly embedding each patch. Subsequently, we embed several ResNet layers <br> in the Vision Transformer (ViT) model, which yield better results than the state-of-the-art convolutional networks. a transformer directly applied to image patch sequences that perform exceptionally well on image classification tasks.We employ the traditional method of appending an extra learnable "classification token" to the series in order to classify the breast cancer <br> images and ascertain if the patient has benign (non-cancerous) or malignant (cancerous) infections.

## The framework of the proposed Vision Transformer model:


![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/breast_cancer_detection-master/PM%20VIT.drawio.png)


## EfficientNet_B3


## MobileNet_V2


## DenseNet-121


## VGG-16



## VGG-19


# Graphical Representation of Image Classification Models:
## Accuracy of Models:
![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/breast_cancer_detection-master/IC_Acc.png)
## Loss of Models:
![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/breast_cancer_detection-master/IC_Loss.png)

# Image Translation Models Used:
## CycleGAN
<br>
The main goal of applying the CycleGAN model is to ensure that the translated and original pictures follow the same cycle. Our suggested model, BCI (Breast Cancer Immunohistochemical) dataset, resolves the paired image-to-image translation issue in CycleGAN. 4870 recruited image sets with a variety of HER2 expression levels are included in the collection of this BCI dataset. A and B are the two picture domains of interest. Two generator systems, G_AB and G_BA, plus two discriminator systems, D_A and D_B, make up the cycleGAN model.
Throughout training, the CycleGAN learns to minimize two different kinds of losses: adversarial loss and cycle consistency loss. The adversarial loss incentivizes the generator to produce realistic-appearing pictures that deceive the discriminator into classifying them as authentic or fraudulent.



## The framework of the proposed CycleGAN model:

![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/CycleGAN/cycleganPM.png)
<br>
## Autoencoders<br>



## PyramidPix2pix


# Graphical Representation of Image-to-Image Translation Models:
## PSNR Values of Models:
![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/PSNR_ig.png)
## SSIM score of Models:
![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/SSIM_ig.png)

# Future Work:
To facilitate more thorough analysis and exploration of histopathological data, future research can also classify the breast histopathological images into different stages of breast cancer and take into account other histopathological image transformations, such as stain normalization, stain augmentation, or cross-modality translation. Beyond H&E translation, we can also use image-to-image translation models to IHC translation.

# Abbreviations:

• CNN : Convolutional Neural Network <br>
• ViT : Vision Transformers <br>
• VGG-16 : 16-layer variant of the Visual Geometry Group (VGG) convolutional
neural network (CNN) architecture.<br>
• VGG-19: VGG-19 stands for the 19-layer variant of the Visual Geometry
Group (VGG) convolutional neural network (CNN) architecture.<br>
• cycleGAN : Cycle-Consistent Generative Adversarial Network <br>
• H & E : Hematoxylin and Eosin <br>
• IHC : Immunohistochemical <br>
• BCI : Breast Cancer Immunohistochemical Dataset <br>
• BreaKHis :Breast Cancer Histopathological Dataset



