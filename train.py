# _*_ coding: utf-8 _*_
# Author: Jielong
# Creation Time: 10/08/2019 00:35
from utils import get_imgs_masks, get_batch_data
from unet.unet_model import UnetModel


def main():
    image_folder = "./data/2d_images/"
    masks_folder = "./data/2d_masks/"
    # # tr_paths, v_paths = get_train_val_paths(image_folder, masks_folder)
    images, labels = get_imgs_masks(image_folder, masks_folder)
    # print(images[0].shape)
    no_samples = images.shape[0]
    batch_size = 4
    n_epochs = 40
    unet = UnetModel()
    unet.train(data_gen=get_batch_data, images=images, labels=labels, n_epochs=n_epochs, n_samples=no_samples)


if __name__ == "__main__":
    main()
