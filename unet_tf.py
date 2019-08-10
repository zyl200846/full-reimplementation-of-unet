# _*_ coding: utf-8 _*_
# Author: Jielong
# Creation Time: 25/07/2019 11:51
import tensorflow as tf
from unet.loss import dice_loss
from unet.metrics import mean_iou
from utils import get_imgs_masks, get_batch_data


x_train = tf.placeholder(dtype=tf.float32, shape=[None, 512, 512, 1], name="x_train")
y_train = tf.placeholder(dtype=tf.float32, shape=[None, 512, 512, 1], name="y_train")
lr = tf.placeholder(dtype=tf.float32)

conv1 = tf.layers.conv2d(inputs=x_train, filters=16, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
conv1 = tf.layers.dropout(inputs=conv1, rate=0.1)
conv1 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
p1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, padding="valid")

conv2 = tf.layers.conv2d(inputs=p1, filters=32, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
conv2 = tf.layers.dropout(inputs=conv2, rate=0.1)
conv2 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
p2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, padding="valid")

conv3 = tf.layers.conv2d(inputs=p2, filters=64, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
conv3 = tf.layers.dropout(inputs=conv3, rate=0.1)
conv3 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
p3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, padding="valid")

conv4 = tf.layers.conv2d(inputs=p3, filters=128, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
conv4 = tf.layers.dropout(inputs=conv4, rate=0.1)
conv4 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
p4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=2, padding="valid")

conv5 = tf.layers.conv2d(inputs=p4, filters=256, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
conv5 = tf.layers.dropout(inputs=conv5, rate=0.2)
conv5 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())

up1 = tf.layers.conv2d_transpose(inputs=conv5, filters=128, kernel_size=2, strides=2,
                                 padding="same", kernel_initializer=tf.initializers.glorot_uniform())
up1 = tf.concat([up1, conv4], axis=3)
conv6 = tf.layers.conv2d(inputs=up1, filters=128, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
conv6 = tf.layers.dropout(inputs=conv6, rate=0.2)
conv6 = tf.layers.conv2d(inputs=conv6, filters=128, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())

up2 = tf.layers.conv2d_transpose(inputs=conv6, filters=64, kernel_size=2, strides=2,
                                 padding="same", kernel_initializer=tf.initializers.glorot_uniform())
up2 = tf.concat([up2, conv3], axis=3)
conv7 = tf.layers.conv2d(inputs=up2, filters=64, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
conv7 = tf.layers.dropout(inputs=conv7, rate=0.2)
conv7 = tf.layers.conv2d(inputs=conv7, filters=64, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())

up3 = tf.layers.conv2d_transpose(inputs=conv7, filters=32, kernel_size=2, strides=2,
                                 padding="same", kernel_initializer=tf.initializers.glorot_uniform())
up3 = tf.concat([up3, conv2], axis=3)
conv8 = tf.layers.conv2d(inputs=up3, filters=32, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
conv8 = tf.layers.dropout(inputs=conv8, rate=0.2)
conv8 = tf.layers.conv2d(inputs=conv8, filters=32, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())

up4 = tf.layers.conv2d_transpose(inputs=conv8, filters=16, kernel_size=2, strides=2,
                                 padding="same", kernel_initializer=tf.initializers.glorot_uniform())
up4 = tf.concat([up4, conv1], axis=3)
conv9 = tf.layers.conv2d(inputs=up4, filters=16, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
conv9 = tf.layers.dropout(inputs=conv9, rate=0.1)
conv9 = tf.layers.conv2d(inputs=conv9, filters=16, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())

conv10 = tf.layers.conv2d(inputs=conv9, filters=1, kernel_size=1, strides=1,
                          kernel_initializer=tf.initializers.glorot_uniform())
logits = tf.nn.sigmoid(conv10)

with tf.name_scope("loss"):
    # loss = cross_entropy(labels=tf.reshape(y_train, [-1, 2]), logits=tf.reshape(logits, [-1, 2]))
    loss = dice_loss(y_true=y_train, y_pred=logits)
    iou = mean_iou(y_true=y_train, y_pred=logits)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    training_op = optimizer.minimize(loss)

image_folder = "./data/2d_images/"
masks_folder = "./data/2d_masks/"
# # tr_paths, v_paths = get_train_val_paths(image_folder, masks_folder)
images, labels = get_imgs_masks(image_folder, masks_folder)
# print(images[0].shape)
batch_size = 4
n_epochs = 40
n_iteration_per_epoch = len(images) // batch_size
# x_batch, y_batch = get_batch_data(images, labels, iter_step=0, batch_size=batch_size)

init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    init.run()
    # for epoch in range(n_epochs):
    #     loss_val, _ = sess.run([loss, training_op], feed_dict={x_train: x_batch, y_train: y_batch, lr: 0.001})
    #     print(loss_val)
    for epoch in range(n_epochs):
        total_loss = 0
        # iou_values = 0
        print("Start training epoch {}".format(epoch + 1))
        for i in range(n_iteration_per_epoch):
            x_batch, y_batch = get_batch_data(images, labels, iter_step=i, batch_size=batch_size)
            loss_val, _ = sess.run([loss, training_op],
                                   feed_dict={x_train: x_batch, y_train: y_batch, lr: 0.0001})
            total_loss += loss_val
            # iou_values += iou_val
        print("Epoch {}, loss: {}".format(epoch + 1, total_loss / n_iteration_per_epoch))

