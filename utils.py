# _*_ coding: utf-8 _*_
# Author: Jielong
# Creation Time: 25/07/2019 10:25
import os
import sys
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from tqdm import tqdm
from random import shuffle, sample
np.set_printoptions(threshold=sys.maxsize)


def standardize(img):
    mean = np.mean(img, axis=0)
    std = np.std(img, axis=0)
    return (img - mean) / std


def min_max_normalize(img):
    """
    Min-max normalization (recommended for image processing)
    :param img: input image
    :return: normalized image
    """
    max_val = np.max(img)
    min_val = np.min(img)
    return (img - min_val) / (max_val - min_val)


# Get training and validation paths
def get_train_val_paths(img_folder, mask_folder, split_ratio=0.8):
    img_names = os.listdir(img_folder)
    print("total images: ", len(img_names))
    mask_names = os.listdir(mask_folder)
    img_paths = [os.path.join(img_folder, name) for name in img_names]
    mask_paths = [os.path.join(mask_folder, name) for name in mask_names]
    no_samples = len(img_paths)
    no_train = np.int(np.ceil(split_ratio * no_samples))
    train_paths = {name: paths_list for name, paths_list in zip(["train_imgs", "train_mask"],
                                                                [img_paths[:no_train], mask_paths[:no_train]])}
    val_paths = {name: paths_list for name, paths_list in zip(["val_imgs", "val_mask"],
                                                              [img_paths[no_train:], mask_paths[no_train:]])}
    return train_paths, val_paths


# Data generator to get single image and mask
def image_generator(img_paths, mask_paths):
    """
    Single data generator
    :param img_paths: images path where data stored
    :param mask_paths: masks images path where masks stored
    :return: image, mask
    """
    for img_path, mask_path in zip(img_paths, mask_paths):
        img = io.imread(img_path)
        img = standardize(img)
        img = np.expand_dims(img, axis=2)
        mask = rgb2gray(io.imread(mask_path)) / 255.
        mask = np.expand_dims(mask, axis=2)
        mask = (mask >= 0.5).astype(np.float32)

        yield img, mask


# Batch data generator 1
def img_batch_generator(img_paths, mask_paths, batch_size=2):
    """
    Batch data generator
    :param img_paths: input path of images
    :param mask_paths: input path of masks
    :param batch_size: batch size to generate images and masks
    :return: batch data with shape [batch_size, height, width, channels]
    """
    while True:
        img_gen = image_generator(img_paths, mask_paths)

        img_batch, mask_batch = [], []
        for img, mask in img_gen:
            img_batch.append(img)
            mask_batch.append(mask)
            if len(img_batch) == batch_size:
                yield np.stack(img_batch, axis=0), np.stack(mask_batch, axis=0)
                img_batch, mask_batch = [], []
        if len(img_batch) != 0:
            yield np.stack(img_batch, axis=0), np.stack(mask_batch, axis=0)


# Batch data generator 2
def mini_batch_data(img_paths, mask_paths, batch_size=1):
    while True:
        train_paths = []
        for img_p, mask_p in zip(img_paths, mask_paths):
            train_paths.append((img_p, mask_p))

        shuffle(train_paths)
        img_mask_paths = sample(train_paths, k=batch_size)
        img_batch, mask_batch = [], []
        for item in img_mask_paths:
            img_path, mask_path = item
            img = np.expand_dims(min_max_normalize(io.imread(img_path)), axis=2)
            img_batch.append(img)
            mask = np.expand_dims(io.imread(mask_path), axis=2)
            mask_batch.append(mask)
        yield np.array(img_batch), np.array(mask_batch)


# Data generator below is mainly used for TensorFlow low and high API Implementation
def get_imgs_masks(images_folder, masks_folder):
    """
    Get whole dataset
    :param images_folder: input images folder
    :param masks_folder: input masks folder
    :return: images and masks with same shape [no_samples, height, width, channels]
    """
    train_ids = next(os.walk(images_folder))[2]
    mask_ids = next(os.walk(masks_folder))[2]
    print(train_ids)
    # print(mask_ids)

    images = np.zeros(shape=(len(train_ids), 512, 512, 1), dtype=np.float32)
    labels = np.zeros(shape=((len(mask_ids)), 512, 512, 1), dtype=np.float32)

    for i, file_name in tqdm(enumerate(train_ids), total=len(train_ids)):
        img_path = images_folder + file_name
        # print(img_path)
        mask_path = masks_folder + file_name
        # print(mask_path)
        img = min_max_normalize(io.imread(img_path))
        new_img = np.expand_dims(img, axis=-1)

        mask = io.imread(mask_path) / 255.
        new_mask = np.expand_dims(mask, axis=-1)
        images[i] = new_img
        labels[i] = new_mask
    return images, labels


def shuffle_data(imgs, masks):
    """
    Shuffle data so as to ensure each training epoch has different inputs
    :param imgs: all images
    :param masks: corresponding all masks
    :return: shuffled images and masks
    """
    permute_idxs = np.random.permutation(len(imgs))
    # print(permute_idxs)
    imgs = imgs[permute_idxs]
    masks = masks[permute_idxs]
    return imgs, masks


def get_batch_data(imgs, masks, iter_step, batch_size=2):
    """
    Function used to get batch images and batch masks
    :param imgs: take the whole data set as function input
    :param masks: take the whole masks set as function input
    :param iter_step: record iteration step to get next batch data
    :param batch_size: batch size for the data
    :return: images batch, masks batch
    """
    if iter_step == 0:
        shuffle_data(imgs, masks)

    step_count = batch_size * iter_step
    return imgs[step_count: (step_count + batch_size)], masks[step_count: (step_count + batch_size)]


if __name__ == "__main__":
    pass
    # imgs_folder = "./data/2d_images/"
    # ms_folder = "./data/2d_masks/"
    # images, labels = get_imgs_masks(imgs_folder, ms_folder)
    # train_imgs, train_masks = get_batch_data(images, labels, iter_step=0, batch_size=10)
    # print(train_imgs[1][:10, :10])
    # print(train_masks[0])

    # image_folder = "./data/2d_images/"
    # masks_folder = "./data/2d_masks/"
    # tr_paths, v_paths = get_train_val_paths(image_folder, masks_folder)
    # train_paths = []
    # for img_p, mask_p in zip(tr_paths["train_imgs"], tr_paths["train_mask"]):
    #     train_paths.append((img_p, mask_p))
    # print(sample(train_paths, k=6))
    # data_gen = mini_batch_data(tr_paths["train_imgs"], tr_paths["train_mask"])
    # img_batch, mask_batch = next(data_gen)
    # print(img_batch.shape)
    # print(img_batch[0])
    # print(mask_batch.shape)
    # print(len(tr_paths["train_imgs"]))
    # print(tr_paths["train_mask"])
    # print(tr_paths["train_imgs"])
    #
    # import matplotlib.pyplot as plt
    # img_gen = image_generator(tr_paths["train_imgs"], tr_paths["train_mask"])
    # img, mask = next(img_gen)
    # print(img.shape)
    # print(img)
    # # plt.imshow(img)
    # # plt.show()
    # print(mask.shape)
    # print(mask)
    # # plt.imshow(mask)
    # # plt.show()
    #
    # img_batch_gen = img_batch_generator(tr_paths["train_imgs"], tr_paths["train_mask"], batch_size=32)
    # steps = len(tr_paths["train_imgs"])
    # for s in range(steps):
    #     batch_img, batch_mask = next(img_batch_gen)
    #     print(batch_img.shape)
    #     print(batch_mask.shape)
