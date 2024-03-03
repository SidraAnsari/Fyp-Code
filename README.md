# Diagnosis and Treatment of Breast Cancer using H&E Image Analysis:

My research uses image classification and translation to diagnose breast cancer disease using hematoxylin and eosin images. Worldwide, breast cancer is the leading cause of death from cancer among women. This study investigates the use of deep learning models to histopathology image-based breast cancer diagnosis and therapy. These photos help identify cellular and tissue components by examining microscopic, color-coded breast cancer tissue slices. First, we used CNN for Image Classification, but this takes a lot of time. Therefore, we moved to Transfer Learning Model, which uses pre-trained models like SOTA model Vision Transformers utilizing BreakHis Dataset. The identification of cellular and tissue components requires the use of H&E pictures. Immunohistochemistry pictures, on the other hand, show the spatial characteristics of particular cell types. In Image-to-Image Translation studies, a suggested model called CycleGAN is utilized to generate/translate immunohistochemistry pictures from Hematoxylin and Eosin images for the treatment of breast cancer.


# Setup:Envs
-Windows<br>
-Python>=3.11<br>
-NVIDIA GPU 

# Framework Used:
-Pytorch

# Datasets Used:
For the Classification working in this study, we employed two benchmark datasets: the BreakHis (Breast Cancer Histopathological) dataset, which is made up of pictures of breast histopathology. The BCI (Breast Cancer Immunohistochemical) dataset, on the other hand, is specifically used for picture production and translation of paired H&E to IHC images and paired H&E with IHC images.
## BreakHis
BreakHis is regarded as the most well-liked and therapeutically significant public histopathology dataset for breast cancer<br>. Using various amplification factors (40X, 100X, 200X, and 400X), 7,909 microscopic stained pictures of surgical biopsy (Sob)breast tumor tissue obtained from 82 individuals are included. It now has 5,429 cancerous and 2,480 benign samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG arrangement). 
- [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
- ![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/Breakhis%20dataset.jpg)

## BCI
To develop therapies for breast cancer, the expression of human epidermal growth factor receptor 2 (HER 2) should be assessed. 4870 recorded picture pairings with various HER2 activity levels are present in this database.
Numerous tests demonstrate that BCI creates additional challenges in the ongoing effort to translate images.  Furthermore, BCI makes it possible for pathology research in the future to quantify HER2 expression using artificial IHC pictures. BCI is a composite dataset that translates images of HER2 expression from H&E to immunohistochemical findings and HE-stained slices to images.
- [BCI](https://bci.grand-challenge.org/dataset/)
 ![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/datasetpreview6.png)
# Methods:
## Image Classification :

The architecture of the pre-trained model is employed as a feature extractor in transfer learning. The model's first layers, which pick up on <br> fundamental visual characteristics like edges and textures, are kept. Because these layers are frozen, the known components are maintained during training without updating their weights.(br>
The breast image <br>dataset is used to adjust and retrain the pre-trained model's later layers, which capture more sophisticated and task-specific properties. This process is known as fine-tuning. Through adjustments to these layers' weights, the model adjusts to the unique features of the cancer picture classification job<br>.

![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/TL%20new.drawio.png)

# Models:
# Image Classification Models Used:
## Vision Transformers
<br>
The foundation of many Transformer blocks is the Vision Transformer (ViT). Using the BreakHis dataset, the ViT breaks the input histopathological picture into fixed-size patches that may be understood as tokens. We next apply a Positional Encoding layer, which linearly embeds each patch, ensuring that the order of our picture patches matches the input. Next, we incorporate many ResNet layers <br> into the Vision Transformer (ViT) model, which achieve superior outcomes in comparison to the most advanced convolutional networks available. A transformer applied directly to picture patch sequences that exhibit excellent performance on image classification challenges. To do the classification of the breast cancer <br> photos, as well as determine if the patient is infected with benign (non-cancerous) or malignant (cancerous) infections, we use the conventional approach of attaching an additional learnable "classification token" to the series.

## The framework of the proposed Vision Transformer model:


![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/breast_cancer_detection-master/PM%20VIT.drawio.png)


## EfficientNet_B3


## MobileNet_V2


## DenseNet-121


## VGG-16



## VGG-19
<br>
VGG-19 is a more extensive and deeper version of VGG-16. VGG19 is a variant of VGG consisting of 19 layers that are 16 convolutional layers, and three fully connected layers,<br> in addition to 5 max-pooling layers and 1 SoftMax layer)\cite{wakili2022classification} \cite{simonyan2014very}.<br> It has a deeper architecture with more parameters and is more computationally expensive,
but it has been shown to achieve better results on some image classification tasks. <br>The accuracy of VGG-19 is 80.3% and 0.87 loss.

# Graphical Representation of Image Classification Models:
## Accuracy of Models:
![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/breast_cancer_detection-master/IC_Acc.png)
## Loss of Models:
![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/breast_cancer_detection-master/IC_Loss.png)

# Image-to-Image Translation(I2I) Models Used:
## CycleGAN
<br>
The main goal of applying the CycleGAN model is to ensure that the translated and original pictures follow the same cycle. Our suggested model, BCI (Breast Cancer Immunohistochemical) dataset, resolves the paired image-to-image translation issue in CycleGAN. 4870 recruited image sets with a variety of HER2 expression levels are included in the collection of this BCI dataset. A and B are the two picture domains of interest. Two generator systems, G_AB and G_BA, plus two discriminator systems, D_A and D_B, make up the cycleGAN model.
Throughout training, the CycleGAN learns to minimize two different kinds of losses: adversarial loss and cycle consistency loss. The adversarial loss incentivizes the generator to produce realistic-appearing pictures that deceive the discriminator into classifying them as authentic or fraudulent.



## The framework of the proposed CycleGAN model:

![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/CycleGAN/cycleganPM.png)
<br>
## Autoencoders<br>
Neural networks using autoencoders are used for dimensionality reduction and unsupervised learning. It is intended to learn an encoding or compressed representation of the input data, from which it will subsequently recreate the original data.(br>

• An encoder and a decoder are the two primary components of autoencoders. An input, usually a high-dimensional data point, is mapped to a lower-dimensional representation known as the latent space or encoding by the encoder.(br>

• Typically, the encoder is made up of several layers that gradually lower the dimensionality of the input data, such as fully connected or convolutional layers.(br>


• In contrast, the encoded representation is given to the decoder, which uses it to reassemble the original input. With layers that progressively increase the dimensionality until the output matches the input, the decoder is a mirror image of the encoder.(br>

• The autoencoder learns to reduce the reconstruction error, which quantifies the variation between the original input and the decoder's output, by feeding it input data during training. Through this training process, the autoencoder is encouraged to identify and pick up useful characteristics from the input data that are required for precise reconstruction. 
<br>The difference in similarity between the input and the output is measured by the loss function. 

• Back-propagation is used to minimize the loss function in order to train the autoencoder. The encoder processes the input data to produce the encoded representation, which is then used by the decoder to reconstruct the input. The loss is calculated as the difference between the original input and the reconstructed output, and the autoencoder's weights are updated by propagating the gradients backward.
![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/Autoencoder/Autoencoder.drawio%20(2).png)


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



