# _*_ coding: utf-8 _*_
# Author: Jielong
# Creation Time: 25/07/2019 10:25
import os
import cv2
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from random import shuffle, sample


def normalize(img):
    mean = np.mean(img, axis=0)
    std = np.std(img, axis=0)
    return (img - mean) / std


def min_max_normalize(img):
    max_val = np.max(img)
    min_val = np.min(img)
    return (img - min_val) / (max_val - min_val)


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


def image_generator(img_paths, mask_paths):

    for img_path, mask_path in zip(img_paths, mask_paths):
        img = io.imread(img_path)
        img = normalize(img)
        img = np.expand_dims(img, axis=2)
        mask = rgb2gray(io.imread(mask_path))
        mask = np.expand_dims(mask, axis=2)
        mask = (mask >= 0.5).astype(np.float32)

        yield img, mask


def img_batch_generator(img_paths, mask_paths, batch_size=2):
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
            img_batch, mask_batch = [], []


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


def data_generator(img_paths, mask_paths, batch_size=8):
    pos = 0

    imgs = np.zeros(shape=(batch_size, 512, 512, 3)).astype(np.float32)
    masks = np.zeros(shape=(batch_size, 512, 512, 1)).astype(np.float32)

    for i in range(pos, pos + batch_size):
        img = cv2.imread(img_paths[i]) / 255.
        mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE) / 255.
        mask = mask.reshape(512, 512, 1)

        imgs[i - pos] = img
        masks[i - pos] = mask

    pos += batch_size
    if (pos + batch_size) > len(img_paths):
        pos = 0
        shuffle(img_paths)
    yield imgs, masks


if __name__ == "__main__":
    # pass
    image_folder = "./data/2d_images/"
    masks_folder = "./data/2d_masks/"
    tr_paths, v_paths = get_train_val_paths(image_folder, masks_folder)
    train_paths = []
    for img_p, mask_p in zip(tr_paths["train_imgs"], tr_paths["train_mask"]):
        train_paths.append((img_p, mask_p))
    print(sample(train_paths, k=6))
    data_gen = mini_batch_data(tr_paths["train_imgs"], tr_paths["train_mask"])
    img_batch, mask_batch = next(data_gen)
    print(img_batch.shape)
    print(img_batch[0])
    print(mask_batch.shape)
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
