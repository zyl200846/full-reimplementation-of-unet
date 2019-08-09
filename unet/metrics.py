# _*_ coding: utf-8 _*_
# Author: Jielong
# Creation Time: 28/07/2019 21:31
import tensorflow as tf
import numpy as np
from keras import backend as K


def dice_coefficient(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_pred_f = tf.cast(tf.greater(tf.reshape(y_pred, [-1]), 0.5), dtype=tf.float32)
    intersection = y_true_f * y_pred_f
    dice_score = 2. * (tf.reduce_sum(intersection) + 1e-9) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-9)
    return dice_score


def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)


def multi_bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.compat.v1.losses.softmax_cross_entropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def binary_bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.compat.v1.losses.sigmoid_cross_entropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


if __name__ == "__main__":
    a = tf.constant([[[1., 0., 1., 0.], [1., 1., 0., 0.]], [[0., 1., 1., 0.], [1., 1., 1., 0.]]], dtype=tf.float32)
    b = tf.constant([[[1., 1., 1., 0.], [0., 1., 0., 0.]], [[1., 1., 0., 0.], [1., 1., 1., 0.]]], dtype=tf.float32)
    c = tf.constant([[[0.78, 0.56, 0.64, 0.23], [0.32, 0.76, 0.23, 0.12]],
                     [[0.67, 0.90, 0.32, 0.11], [0.87, 0.75, 0.98, 0.08]]])
    d = dice_coefficient(a, b)
    e = dice_loss(a, b)
    f = binary_bce_dice_loss(a, c)
    g = multi_bce_dice_loss(a, c)

    with tf.compat.v1.Session() as sess:
        print(sess.run(d))
        print(sess.run(e))
        print(sess.run(f))
        print(sess.run(g))

