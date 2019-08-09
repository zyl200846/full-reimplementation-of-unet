# _*_ coding: utf-8 _*_
# Author: Jielong
# Creation Time: 29/07/2019 16:44
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from skimage.transform import resize
from skimage.io import imshow
from utils import img_batch_generator, get_train_val_paths
from unet.metrics import mean_iou, dice_loss


image_folder = "./data/2d_images/"
masks_folder = "./data/2d_masks/"
tr_paths, v_paths = get_train_val_paths(image_folder, masks_folder)
val_gen = img_batch_generator(v_paths["val_imgs"], v_paths["val_mask"], batch_size=2)
img, mask = next(val_gen)
print(img[0].shape)
model = load_model("unet_model.h5", custom_objects={"mean_iou": mean_iou, "dice_loss": dice_loss})
preds_val = model.predict(img)
preds_val = (preds_val >= 0.5).astype(np.uint8)
imshow(np.squeeze(preds_val[1]))
plt.show()
imshow(np.squeeze(mask[1]))
plt.show()
