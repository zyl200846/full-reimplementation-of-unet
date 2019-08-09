import os
from tqdm import tqdm
import numpy as np
from skimage import io
from utils import min_max_normalize
import sys
np.set_printoptions(threshold=sys.maxsize)


def get_imgs_masks(images_folder, masks_folder):
    train_ids = next(os.walk(images_folder))[2]
    mask_ids = next(os.walk(masks_folder))[2]
    # print(train_ids)
    # print(mask_ids)

    images = np.zeros(shape=(len(train_ids), 512, 512, 1), dtype=np.uint8)
    labels = np.zeros(shape=((len(mask_ids)), 512, 512, 1), dtype=np.uint8)

    for i, file_name in tqdm(enumerate(train_ids), total=len(train_ids)):
        img_path = images_folder + file_name
        mask_path = masks_folder + file_name
        img = min_max_normalize(io.imread(img_path))
        new_img = np.reshape(img, newshape=(img.shape[0], img.shape[1], 1))

        mask = io.imread(mask_path) / 255
        new_mask = np.reshape(mask, newshape=(mask.shape[0], mask.shape[1], 1))
        images[i] = new_img
        labels[i] = new_mask
    return images, labels


def shuffle_data(imgs, masks):
    permute_idxs = np.random.permutation(len(imgs))
    # print(permute_idxs)
    imgs = imgs[permute_idxs]
    masks = masks[permute_idxs]
    return imgs, masks


def get_batch_data(imgs, masks, iter_step, batch_size=2):
    if iter_step == 0:
        shuffle_data(imgs, masks)

    step_count = batch_size * iter_step
    return imgs[step_count: (step_count + batch_size)], masks[step_count: (step_count + batch_size)]


if __name__ == "__main__":
    imgs_folder = "./data/2d_images/"
    ms_folder = "./data/2d_masks/"
    images, labels = get_imgs_masks(imgs_folder, ms_folder)
    train_imgs, train_masks = get_batch_data(images, labels, iter_step=0, batch_size=10)
    print(train_imgs.shape)
    print(train_masks[0])
