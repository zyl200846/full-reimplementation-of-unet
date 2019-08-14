# _*_ coding: utf-8 _*_
# Author: Jielong
# Creation Time: 10/08/2019 00:38
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.io import imshow, imsave
from skimage import img_as_int
from utils import get_batch_data, get_imgs_masks


with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph("./models/tf_model.ckpt.meta")
    # saver.restore(sess, tf.compat.v1.train.latest_checkpoint("./models/"))
    saver.restore(sess, "./models/tf_model.ckpt")
    graph = tf.compat.v1.get_default_graph()
    # x_test = graph.compat.v1.get_tensor_by_name("x_input:0")
    x_test = graph.get_operation_by_name("x_input").outputs[0]
    y_test = tf.compat.v1.get_collection("network_architecture")[0]

    image_folder = "./data/2d_images/"
    masks_folder = "./data/2d_masks/"
    images, labels = get_imgs_masks(image_folder, masks_folder)
    batch_size = 2
    no_samples = images.shape[0]
    n_iterations = no_samples // batch_size
    predictions = []
    for i in range(n_iterations):
        x_batch, y_batch = get_batch_data(images, labels, iter_step=i, batch_size=2)
        prediction = sess.run(y_test, feed_dict={x_test: x_batch})
        print(prediction.shape)
        predictions.append(prediction)

print("shape of images: ", np.array(predictions).shape)
# preds_val = predictions[0]
# print("halo shape: ", preds_val.shape)
# preds_val = (preds_val >= 0.5).astype(np.uint8)
# print(preds_val[0].shape)
# print(np.squeeze(preds_val[1]).shape)
# # imsave(fname="./pred.png", arr=img_as_int(np.squeeze(preds_val[1])))
# imshow(np.squeeze(preds_val[1]))
# plt.show()
