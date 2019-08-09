# _*_ coding: utf-8 _*_
# Author: Jielong
# Creation Time: 25/07/2019 11:51
import tensorflow as tf
from utils import get_train_val_paths, mini_batch_data, img_batch_generator
from unet.metrics import dice_loss, mean_iou
from test import get_imgs_masks, get_batch_data


def conv2d(X, filters, name, k_size=3, padding="SAME"):
    with tf.name_scope("conv_block"):
        conv = tf.layers.conv2d(inputs=X, filters=filters, kernel_size=k_size,
                                strides=1, padding=padding, activation=tf.nn.relu, name=name)
        return conv


def deconv2d(X, filters, name, k_size=2, padding="SAME"):
    with tf.name_scope("deconv_block"):
        deconv = tf.layers.conv2d_transpose(inputs=X, filters=filters, kernel_size=k_size,
                                            strides=2, padding=padding,
                                            kernel_initializer=tf.compat.v1.initializers.glorot_uniform(), name=name)
        return deconv


def max_pool(X, name, k_size=2, strides=2, padding="VALID"):
    with tf.name_scope("max_pool"):
        return tf.nn.max_pool2d(input=X, ksize=k_size, strides=strides, padding=padding, name=name)


def drop_out(X, rate=0.1):
    with tf.name_scope("dropout"):
        return tf.nn.dropout(X, rate=rate)


with tf.name_scope("inputs"):
    X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 512, 512, 1], name="X")
    y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 512, 512, 1], name="y")

# Model Creation
# Contract path
conv1 = conv2d(X, filters=64, name="conv1")
conv1 = drop_out(conv1, rate=0.1)
conv2 = conv2d(conv1, filters=64, name="conv2")
max_pool1 = max_pool(conv2, name="max_pool1")

conv3 = conv2d(max_pool1, filters=128, name="conv3")
conv3 = drop_out(conv3, rate=0.1)
conv4 = conv2d(conv3, filters=128, name="conv4")
max_pool2 = max_pool(conv4, name="max_pool2")

conv5 = conv2d(max_pool2, filters=256, name="conv5")
conv5 = drop_out(conv5, rate=0.1)
conv6 = conv2d(conv5, filters=256, name="conv6")
max_pool3 = max_pool(conv6, name="max_pool3")

conv7 = conv2d(max_pool3, filters=512, name="conv7")
conv7 = drop_out(conv7, rate=0.2)
conv8 = conv2d(conv7, filters=512, name="conv8")

# Expansive path
deconv1 = deconv2d(conv8, filters=256, name="deconv1")
concat_layer1 = tf.concat([conv6, deconv1], axis=3)
conv9 = conv2d(concat_layer1, filters=256, name="conv9")
conv9 = drop_out(conv9, rate=0.1)
conv10 = conv2d(conv9, filters=256, name="conv10")

deconv2 = deconv2d(conv10, filters=128, name="deconv2")
concat_layer2 = tf.concat([conv4, deconv2], axis=3)
conv11 = conv2d(concat_layer2, filters=128, name="conv11")
conv11 = drop_out(conv11, rate=0.1)
conv12 = conv2d(conv11, filters=128, name="conv12")

deconv3 = deconv2d(conv12, filters=64, name="deconv3")
concat_layer3 = tf.concat([conv2, deconv3], axis=3)
conv13 = conv2d(concat_layer3, filters=64, name="conv13")
conv13 = drop_out(conv13, rate=0.2)
conv14 = conv2d(conv13, filters=64, name="conv14")

logits = tf.layers.conv2d(conv14, filters=1, kernel_size=1, name="conv_final")
logits = tf.nn.sigmoid(logits)
print("The final output shape", logits.shape)

with tf.name_scope("training_op"):
    loss = dice_loss(y_true=y, y_pred=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    training_op = optimizer.minimize(loss)

with tf.name_scope("init"):
    init = tf.compat.v1.global_variables_initializer()

image_folder = "./data/2d_images/"
masks_folder = "./data/2d_masks/"
# tr_paths, v_paths = get_train_val_paths(image_folder, masks_folder)
images, labels = get_imgs_masks(image_folder, masks_folder)
print(images.shape)
batch_size = 2
n_epochs = 20
n_iteration_per_epoch = len(images) // batch_size       # len(tr_paths["train_imgs"]) // batch_size


with tf.compat.v1.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        total_loss = 0
        print("Start training epoch {}".format(epoch + 1))
        for i in range(n_iteration_per_epoch):
            x_batch, y_batch = get_batch_data(images, labels, iter_step=i, batch_size=2)
            loss_val, _ = sess.run([loss, training_op], feed_dict={X: x_batch, y: y_batch})
            total_loss += loss_val
        print("Epoch {}, loss: {}".format(epoch + 1, total_loss / n_iteration_per_epoch))
#         total_loss = 0
#         print("Start training epoch {}".format(epoch + 1))
#         data_gen = img_batch_generator(tr_paths["train_imgs"], tr_paths["train_mask"], batch_size=batch_size)
#         for iteration in range(n_iteration_per_epoch):
#             x_batch, y_batch = next(data_gen)
#             sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
#             loss_val = sess.run(loss, feed_dict={X: x_batch, y: y_batch})
#             total_loss += loss_val
#         print("Epoch {} Training loss is: {}".format(epoch + 1, total_loss / n_iteration_per_epoch))
