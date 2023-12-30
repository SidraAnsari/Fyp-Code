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

## Datasets Used:

- [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
- [BCI](https://bci.grand-challenge.org/dataset/)

## Framework Used:
Pytorch

# Models:
## Image Classification Models Used:
-Vision Transformers<br>
-Densenet121<br>
-Vgg-16<br>
-Vgg-19<br>
-Mobilenetv2<br>
-Efficientnetb3

## Image Translation Models Used:
# Autoencoders<br>
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

• The autoencoder is trained by minimizing the loss function through back-propagation.<br> The input data is passed through the encoder to obtain the encoded representation, and then the decoder <br> reconstructs the input from the encoded representation The difference between the original input and the reconstructed output <br> is used to compute the loss, and the <br>gradients are propagated backward to update the weights of the autoencoder.<br>

-CycleGan<br>
-PyramidPix2pix

## The framework of the proposed pyramid pix2pix model:

![Alt text](https://github.com/SidraAnsari/Fyp-Code/blob/main/Pyramidpix2pix%20framework.png)



