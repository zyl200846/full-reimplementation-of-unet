# full-reimplemnetation-of-unet
This is a full implementation of UNet using **TensorFlow with low level API** and **high level API** as well as **Keras**.

If you need to read detailed description of UNet architecture, please refer to journal article which was proposed by Ronneberger et al. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

## Differences compared to original paper
This code can be trained to segmenting arbitrary sizes/shapes of images. However, one should be noticed that **there are some differences between this code and the original one descrbed by the UNet proposer**:
* The original paper used "relu" as activation function followed by every convolution, while I used "elu" here so as to avoid dying relu caused by relu operation; (Notice that this might not necessarily happen in UNet architecture if one trains the model with certain depth)
* Using "Adam" instead of "SGD" for better convergence. if you want to train the model with SGD you can change it in the code;
* Using "same" padding instead of "valid" padding so as to have the final output with same size as input image;
* Adding Batch Normalization to accelerate training, reduce internal covariant shift, allow use of saturating non-linearities and higher learning rates, [refer here to check why we use BN in deep CNN](https://gist.github.com/shagunsodhani/4441216a298df0fe6ab0);
* As this code is tested on a binary dataset, so I used "sigmoid" as final activation function instead of pixel-wise softmax;
* Besides, I also did not calculate the loss using cross entropy. Instead, dice loss is applied.

## How to use the code
##### Structure of the repository

  


## Results
##### Segmentation results with TF Low Level API


##### Segmentation results with TF High Level API


##### Segementation results with Keras
![ground truth]()

## Python Libraries Required to Run the Code
* tensorflow-gpu==1.14
* keras==2.2.4
* scikit-image==0.15.0
* tqdm==4.32.1
* numpy==1.16.4

Note: it is bettern to create a virtual environment in case there are conflicts between different projects.
