# Diagnosis and Treatment of Breast Cancer through Haematoxylin and Eosin Image Analysis

My Research based on Breast Cancer Disease diagnosis and treatment of Haematoxylin & Eosin Images with the help of Image Classification and Translation.
Breast cancer is the driving cause of cancer-related
passings among women around the world. This research explores
the use of deep learning models in Breast cancer diagnosis and
treatment using Hematoxylin and Eosin recolored histopatho-
logical images. These images are used to
examine breast cancer tissue sections which are
microscopic and color-coded, aid in identifying cellular and tissue
components.
Image Classification performed on CNN and Transfer Learning Models such as Vision Transformers using BreakHis Dataset.
Image Generation/Translation performed on Autoencoders,Pyramidpix2pix and CycleGans Models using Breast Cancer Immunohistochemical Dataset.

# Setup:Envs
-Windows<br>
-Python>=3.11<br>
-NVIDIA GPU 

# Framework Used:
-Pytorch

# Datasets Used:

- [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

  - ![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/Breakhis%20dataset.jpg)
  
- [BCI](https://bci.grand-challenge.org/dataset/)
  
  ![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/datasetpreview6.png)

# Models:
# Image Classification Models Used:
## Vision Transformers
<br>
Vision Transformer (ViT) is the backbone of multiple Transformer blocks. The ViT processes the input histopathology image using the BreakHis dataset and splits the image into fixed-size patches,<br> which are understandable as tokens. We add a Positional Encoding layer that linearly embeds each patch, so our image patches are arranged in the same order as the input. Then, we add a few ResNet layers <br> to the Vision Transformer (ViT) model that attain excellent results compared to the state-of-the-art convolutional networks. A transformer implemented directly to sequences of image patches that perform <br> very well on image classification tasks. We employ the standard technique of appending an additional learnable "classification token" to the series to carry out the classification of the breast cancer <br> images, as whether the patient is infected with benign (non-cancerous) and malignant(cancerous)\cite{dosovitskiy2021image}. It is the best-proposed model among all our classification models, with the highest accuracy of 92.5%. 

## Image Classification using Vision Transformer model using Transfer Learning:

![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/vit%20#model%20image.jpg)

## The framework of the proposed Vision Transformer model:

![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/breast_cancer_detection-master/PM%20VIT.drawio.png)

-Densenet121<br>
-Vgg-16<br>
-Vgg-19<br>
-Mobilenetv2<br>
-Efficientnetb3

# Image Translation Models Used:
## CycleGan
<br>
The key idea behind implementing the CycleGAN model is to enforce cycle consistency between the translated and original images. The BCI (Breast Cancer Immunohistochemical) dataset solves <br> the problem of paired image-to-image translation in CycleGAN, so it is our proposed model. This BCI dataset contains 4870 recruited picture sets with a range of HER2 expression levels included in the <br> collection. The two image domains of interest are A and B. The model of cycleGAN consists of two generator systems, G\_AB and G\_BA, and two discriminator systems, D\_A and D\_B.<br>
The CycleGAN learns to minimize two types of losses during training: adversarial loss and cycle consistency loss. The adversarial loss encourages the generator to generate realistic-looking <br> images that fool the discriminator into identifying real or fake images. Generator G\_AB takes an input image from A and tries to generate a realistic image in  B that tricks discriminator D\_B with <br> mapping G\_AB: A → B. So in the same way also, generator G\_BA creates an image and tries to trick discriminator D\_A whereas D\_A points to recognize between images in domain A and interpreted images G\_BA, D\_B points <br> to discriminate between
images in domain B  and G\_AB\cite{jose2021generative}\cite{zhu2020unpaired}. The generator takes an image from the source domain (H\&E) and tries <br>
to transform it into an image that resembles the target domain (IHC). On the other hand, the discriminator is responsible for distinguishing between the generated and <br> original images from the target domain.
By iteratively optimizing these losses, the CycleGAN can learn to map H\&E images to IHC images and vice versa.<br> This approach is instrumental in medical image analysis.
Once trained, <br>the CycleGAN can convert unseen H\&E images to IHC images by feeding them into the generator.



## The framework of the proposed CycleGAN model:

![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/CycleGAN/cycleganPM.png)
<br>
## Autoencoders<br>
Autoencoder is a type of neural network that is used for unsupervised learning and <br> dimensionality reduction. It is designed to learn a compressed representation or encoding of <br>the input data and then reconstruct the original data from this compressed representation.<br>

• Autoencoders consist of two main parts: an encoder and a decoder. The <br>
encoder takes an input, typically a high-dimensional data point, and maps it <br>
to a lower-dimensional representation, called the latent space or encoding.<br>

• The encoder typically consists of multiple layers, such as fully connected or <br>
convolutional layers, that gradually reduce the dimensionality of the input <br>
data.<br>

• The decoder, on the other hand, takes the encoded representation and <br>
reconstruct the original input from it. The decoder is a mirror image of the encoder,<br>
with layers that gradually increase the dimensionality until the output matches <br>
the input.<br>

• During training, the autoencoder is fed with the input data and learns to <br>
minimize the reconstruction error, which measures the difference between the <br>
original input and the output of the decoder. This training process encourages <br>
the autoencoder to capture and learn meaningful features from the input data <br>
that are necessary for accurate reconstruction. The compressed representation in <br>
the latent space serves as a bottleneck that forces the autoencoder to <br>
capture the most salient information from the data. The loss function measures the <br>dissimilarity between the input and the output. The choice of loss function depends on <br>
the type of data and the desired properties of the autoencoder. Mean Squared Error (MSE) is <br>commonly used for continuous data,while Binary Cross-Entropy (BCE) is often used for <br>
binary data.

• The autoencoder is trained by minimizing the loss function through back-propagation.<br> The input data is passed through the encoder to obtain the encoded representation, and then the decoder reconstructs the input from the encoded representation The difference <br>between the original input and the reconstructed output is used to compute the loss, and the gradients are propagated backward to update the weights of the autoencoder.
<br>
![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/Autoencoder/Autoencoder.drawio%20(2).png)


-PyramidPix2pix

## Abbreviations

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



