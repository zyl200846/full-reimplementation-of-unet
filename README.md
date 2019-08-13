# Full Re-implementation of UNet for Medical Image Segmentation
This is a full implementation of UNet using **TensorFlow with low level API** and **high level API** as well as **Keras**. This repository is still working in progress, things may be changed over time.

If you need to read detailed description of UNet architecture, please refer to journal article which was proposed by Ronneberger et al. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

## Differences compared to original paper
This code can be trained to segmenting arbitrary sizes/shapes of images. However, one should be noticed that **there are some differences between this code and the original one described by the UNet proposer**:
* The original paper used "relu" as activation function followed by every convolution, while I used "elu" here so as to avoid dying relu caused by relu operation; (Notice that this might not necessarily happen in UNet architecture if one trains the model with certain depth)
* Using "Adam" optimizer instead of "SGD" for better convergence. If you want to train the model with SGD you can change it in the code;
* Using "same" padding instead of "valid" padding so as to make the final output have the same size as input images;
* Adding "Batch Normalization" to accelerate training, reduce internal covariant shift, allow use of saturating non-linearities and higher learning rates, [refer here to check why we use BN in deep CNN](https://gist.github.com/shagunsodhani/4441216a298df0fe6ab0);
* As this code is tested on a binary dataset, so I used "sigmoid" as final activation function to generate output instead of pixel-wise softmax mentioned in the paper;
* Besides, I did not calculate the loss using cross entropy. Instead, dice-loss was applied.

## How to use the code
##### Description of files and directories
- data (directory): please put your own images and masks data here, but you should note that you might need to change the code in order to read data correctly.
- unet (directory): this is the implementation code of tensorflow low level api
  - "loss.py": python file that defines loss functions
  - "unet_components": python file that defines convolution op, pooling op, deconvolution op, weights and biases initialization
  - "unet_model": defines whole process of how to train UNet architecture
  - "predict.py": load trained model and then use it to predict validation/test images
  - "metrics.py": define the function of "intersection over union" for evaluating the segmentation results
- "utils.py": defines functions used to get images and masks paths and dataloader function
- "train.py": run this file to train TF low level api implementation of UNet
- "unet_tf.py": tensorflow high level api implementation of UNet, run this file to directly train the model
- "unet_keras.py": keras implementation of UNet, run this file to directly train the model
- "predict_keras.py": used to predict images using trained model by Keras

##### Tips for modifying hyper-parameters to successfully run the code
- If you want to use this code to train on your own dataset quickly, you can directly modify corresponding hyperparameters in "unet_keras.py" and "unet_tf.py" with correct dataloader;
- Otherwise, if you would like to train it using tf low level api, changes can be mainly made to **\_\_init\_\_** part in "unet_model.py";

## Results
### Segmentation results with TF Low Level API

##### Training loss and mIOU
<p align="center">
	<img src="https://github.com/JielongZ/full-reimplementation-of-unet/blob/master/images/save_training_summary_Dice_Loss.svg" width="300" height="280">
	<img src="https://github.com/JielongZ/full-reimplementation-of-unet/blob/master/images/save_training_summary_IOU.svg" width="300" height="280">
</p>

### Segmentation results with TF High Level API

##### Training loss and mIOU

### Segementation results with Keras
##### The left image is the ground truth while the right image is the segmentation result.
<p align="center">
	<img src="https://github.com/JielongZ/full-reimplemnetation-of-unet/blob/master/images/Ground%20Truth.png" width="300" height="280">
	<img src="https://github.com/JielongZ/full-reimplemnetation-of-unet/blob/master/images/predictions.png" width="300" height="280">
</p>

##### Training loss and mIOU

## Python Libraries Required to Run the Code
* tensorflow-gpu==1.14
* keras==2.2.4
* scikit-image==0.15.0
* tqdm==4.32.1
* numpy==1.16.4

Note: it is better to create a virtual environment in case there are conflicts between different projects. Moreover, this code has been successfully run on Windows and trained via Nvidia GTX1060, GTX1080 and GTX1080Ti with variant batch size.
