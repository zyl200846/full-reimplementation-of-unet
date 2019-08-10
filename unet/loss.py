# _*_ coding: utf-8 _*_
# Author: Jielong
# Creation Time: 25/07/2019 10:25
import tensorflow as tf


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


def generalized_dice_loss(labels, logits):
    smooth = 1e-15
    logits = tf.nn.softmax(logits)
    weights = 1. / (tf.reduce_mean(labels, axis=[0, 1, 2]) ** 2 + smooth)

    numerator = tf.reduce_sum(labels * logits, axis=[0, 1, 2])
    numerator = tf.reduce_sum(weights * numerator)

    denominator = tf.reduce_sum(labels + logits, axis=[0, 1, 2])
    denominator = tf.reduce_sum(weights * denominator)

    loss = 1. - 2. * (numerator + smooth) / (denominator + smooth)
    return loss


def pixel_wise_softmax(output_featmap):
    with tf.name_scope("pixel_wise_softmax"):
        max_val = tf.reduce_max(output_featmap, axis=3, keepdims=True)
        exponential_featmap = tf.math.exp(output_featmap - max_val)
        normalize_factor = tf.reduce_sum(exponential_featmap, axis=3, keepdims=True)
        return exponential_featmap / normalize_factor


def cross_entropy(labels, logits):
    return -tf.reduce_mean(labels * tf.math.log(tf.clip_by_value(logits, 1e-10, 1.0)), name="cross_entropy")


def multi_bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.compat.v1.losses.softmax_cross_entropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def binary_bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.compat.v1.losses.sigmoid_cross_entropy(y_true, y_pred) + dice_loss(y_true, y_pred)


if __name__ == "__main__":
    a = tf.constant([[[1., 0., 1., 0.], [1., 1., 0., 0.]], [[0., 1., 1., 0.], [1., 1., 1., 0.]]], dtype=tf.float32)
    b = tf.constant([[[1., 1., 1., 0.], [0., 1., 0., 0.]], [[1., 1., 0., 0.], [1., 1., 1., 0.]]], dtype=tf.float32)
    c = tf.constant([[[0.78, 0.56, 0.64, 0.23], [0.32, 0.76, 0.23, 0.12]],
                     [[0.67, 0.90, 0.32, 0.11], [0.87, 0.75, 0.98, 0.08]]])
    d = dice_coefficient(a, b)
    e = dice_loss(a, c)
    f = binary_bce_dice_loss(a, c)
    g = multi_bce_dice_loss(a, c)

    with tf.compat.v1.Session() as sess:
        print(sess.run(d))
        print(sess.run(e))
        print(sess.run(f))
        print(sess.run(g))
