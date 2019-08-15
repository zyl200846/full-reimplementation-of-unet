# _*_ coding: utf-8 _*_
# Author: Jielong
# Creation Time: 29/07/2019 16:44
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.io import imshow, imsave
from skimage import img_as_int
from utils import img_batch_generator, get_train_val_paths
from unet.loss import dice_loss
from unet.metrics import mean_iou


image_folder = "./data/2d_images/"
masks_folder = "./data/2d_masks/"
tr_paths, v_paths = get_train_val_paths(image_folder, masks_folder)
val_gen = img_batch_generator(v_paths["val_imgs"], v_paths["val_mask"], batch_size=2)
# img, mask = next(val_gen)
# print(img[1].shape)
model = load_model("./models/unet_model.h5", custom_objects={"mean_iou": mean_iou, "dice_loss": dice_loss})
# preds_val = model.predict(img)
# preds_val = (preds_val >= 0.5).astype(np.uint32)
# imshow(np.squeeze(preds_val[1]))
# plt.show()
# imshow(np.squeeze(mask[1]))
# plt.show()

for i, (batch_imgs, batch_masks) in enumerate(val_gen):
    batch_preds = model.predict(batch_imgs)
    batch_preds = (batch_preds >= 0.5).astype(np.uint8)
    masks = (batch_masks >= 0.5).astype(np.uint8)
    for j in range(batch_imgs.shape[0]):
        pred_arr = img_as_int(np.squeeze(batch_preds[j]))
        mask_arr = img_as_int(np.squeeze(masks[j]))
        original_img_arr = np.squeeze(batch_imgs[j])
        imsave(fname="./segments/keras/{}_{}_seg.png", arr=pred_arr)
        imsave(fname="./segments/keras/{}_{}_mask.png", arr=mask_arr)
        imsave(fname="./segments/keras/{}_{}_img.png", arr=original_img_arr)
