# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 01/08/2019 14:41
import tensorflow as tf
from unet.unet_components import weight_init, bias_init, conv2d, max_pool, deconv2d, crop_and_copy
from unet.metrics import dice_coefficient, dice_loss, binary_bce_dice_loss, mean_iou
from utils import get_train_val_paths, img_batch_generator, mini_batch_data


class UnetModel(object):

    def __init__(self, learning_rate=0.0005, model_depth=4, conv_ops=2, k_size=3,
                 pool_size=2, feature_maps_root=16, dropout_rate=0.2):
        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 512, 512, 1], name="x_input")
        self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 512, 512, 1], name="y_label")

        self.learning_rate = learning_rate
        self.model_depth = model_depth
        self.conv_ops = conv_ops
        self.k_size = k_size
        self.pool_size = pool_size
        self.feat_maps_root = feature_maps_root
        self.dropout_rate = dropout_rate

    def build_model(self, X):
        # Extraction Path
        convs_rslt = []
        for depth in range(self.model_depth):
            print("the down level is: ", depth)
            feature_maps = 2 ** depth * self.feat_maps_root
            print("Feature maps = ", feature_maps)
            stddev = tf.sqrt(2 / (self.k_size ** 2 * feature_maps))

            convs_temp = []
            conv = convs_rslt[depth - 1][1] if convs_rslt else X
            for conv_op in range(self.conv_ops):
                if depth == 0:
                    input_feat_channels = tf.shape(X)[3]
                else:
                    input_feat_channels = tf.shape(convs_rslt[depth - 1][0][1])[3]
                W = weight_init(w_shape=(self.k_size, self.k_size, input_feat_channels, feature_maps), std=stddev)
                b = bias_init(value=0.1, shape=[feature_maps], name="bias_{0}_{1}".format(depth, conv_op))
                if depth == 0:
                    conv = conv2d(X=conv, W=W, b=b, rate=0.2)
                    conv = tf.nn.elu(features=conv)
                    print("After convolution: ", conv.shape)
                    convs_temp.append(conv)
                else:
                    conv = conv2d(X=conv, W=W, b=b, rate=0.2)
                    conv = tf.nn.elu(conv)
                    print("After convolution: ", conv.shape)
                    convs_temp.append(conv)

            if depth == self.model_depth - 1:
                rslt_for_deconv = convs_temp[1]
                print("After downsampling, the shape is: ", rslt_for_deconv.shape)
                convs_rslt.append(convs_temp)
            else:
                pool = max_pool(convs_temp[1])
                # X = pool
                print("After max pooling: ", pool.shape)
                print("\n")
                convs_rslt.append(
                    (convs_temp, pool))  # conv_rslt[0][1], conv_rslt[1][1], conv_rslt[2][1], conv_rslt[3][1]

        # Expansive Path
        print("\n")
        deconv_rslt = []
        step = -1
        for depth in range(self.model_depth - 2, -1, -1):
            print("The up level is: ", depth)
            feature_maps = 2 ** (depth + 1) * self.feat_maps_root
            print("the up feature maps are: ", feature_maps)
            stddev = tf.sqrt(2 / (self.k_size ** 2 * feature_maps))
            # conv = X
            conv = convs_rslt[-1][1] if depth == self.model_depth - 2 else deconv_rslt[step][1]
            W_d = weight_init(w_shape=[self.pool_size, self.pool_size, feature_maps // 2, feature_maps], std=stddev)
            # print(W_d.shape)
            b_d = bias_init(value=0.1, shape=[feature_maps // 2], name="up_bias_{0}".format(depth))
            # print(b_d.shape)
            deconv = deconv2d(conv, W=W_d, strides=self.pool_size)
            concat_deconv_conv = crop_and_copy(convs_rslt[depth][0][1], deconv)
            print("After deconv: ", deconv.shape)
            print("concat result: ", concat_deconv_conv.shape)

            # X = concat_deconv_conv
            convs_temp = []
            for conv_op in range(self.conv_ops):
                b = bias_init(value=0.1, shape=[feature_maps // 2], name="up_bias_{0}_{1}".format(depth, conv_op))
                if conv_op == 0:
                    W = weight_init(w_shape=[self.k_size, self.k_size, feature_maps, feature_maps // 2], std=stddev)
                    conv = conv2d(X=concat_deconv_conv, W=W, b=b, rate=0.2)
                    conv = tf.nn.elu(features=conv)
                    print("Shape of data after upsamling and convolution: ", conv.shape)
                else:
                    W = weight_init(w_shape=[self.k_size, self.k_size, feature_maps // 2, feature_maps // 2], std=stddev)
                    conv = conv2d(X=convs_temp[0], W=W, b=b, rate=0.2)
                    conv = tf.nn.elu(features=conv)
                    print("Shape of data after upsamling and convolution: ", conv.shape)
                # X = conv
                convs_temp.append(conv)
            # print("The length is: ", len(convs_temp))
            deconv_rslt.append(convs_temp)
            step += 1
            print("\n")

        with tf.name_scope("final_output"):
            stddev = tf.sqrt(2 / (self.k_size ** 2 * self.feat_maps_root))
            W = weight_init(w_shape=(1, 1, self.feat_maps_root, 1), std=stddev)
            b = bias_init(value=0.1, shape=[1], name="final_out_bias")
            output = conv2d(X=deconv_rslt[-1][1], W=W, b=b, rate=0.1)
            output = tf.nn.sigmoid(output)
            print("final output shape", output.shape)

        return output

    def train(self, data_gen, n_epochs, n_samples):
        logits = self.build_model(self.x)

        with tf.name_scope("training_op"):
            loss = self.get_loss(y_true=self.y, y_preds=logits, loss_mode="cross_entropy")
            # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            optimizer = self.get_optimizer(opt="Adam")
            training_op = optimizer.minimize(loss)
            var_grad = tf.gradients(loss, [logits])[0]

        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            init.run()
            training_steps_per_epoch = n_samples // batch_size
            for epoch in range(n_epochs):
                total_loss = 0
                for step in range(training_steps_per_epoch):
                    x_batch, y_batch = next(data_gen)
                    lg_loss, _ = sess.run([loss, training_op], feed_dict={self.x: x_batch, self.y: y_batch})
                    # var_gra_val = sess.run(var_grad, feed_dict={self.x: x_batch, self.y: y_batch})
                    print("Training step {:}, the step loss: {:.4f}".format(step, lg_loss))
                    # print("Gradients: ", var_gra_val)
                    total_loss += lg_loss
                print("Epoch: {:}, Average loss: {:.4f}".format((epoch + 1), (total_loss / training_steps_per_epoch)))

    @staticmethod
    def get_loss(y_true, y_preds, loss_mode="dice_loss"):
        with tf.name_scope("loss"):
            if loss_mode == "dice_loss":
                loss = dice_loss(y_true=y_true, y_pred=y_preds)
            elif loss_mode == "cross_entropy":
                y_true_flattened = tf.reshape(y_true, [-1])
                y_preds_flattened = tf.reshape(y_preds, [-1])
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_flattened,
                                                                              logits=y_preds_flattened))
            # elif loss_mode == "softmax_with_cross_entropy":
            #     y_preds_reshaped = tf.reshape(y_preds, [-1, 1])
            #     y_true_reshaped = tf.reshape(y_true, [-1])
            #     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_reshaped,
            #                                                                          logits=y_preds_reshaped))
            else:
                raise ValueError("Unknown Cost Function: %s" % loss_mode)
        return tf.convert_to_tensor(loss)

    def get_optimizer(self, opt="Adam"):
        with tf.name_scope("optimizer"):
            if opt == "Adam":
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif opt == "SGD":
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        return optimizer

    def predict(self, test_data, model_path):
        init = tf.compat.v1.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            self.load_model(sess, model_path)
            predictions = sess.run(feed_dict={self.x: test_data})
        return predictions

    @staticmethod
    def dump_model(sess, model_path):
        saver = tf.train.Saver()
        saved_path = saver.save(sess, model_path)
        print("Model has been saved into disk at path: %s" % saved_path)
        return saved_path

    @staticmethod
    def load_model(sess, model_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("Model Loaded!")


if __name__ == "__main__":
    image_folder = "../data/2d_images/"
    masks_folder = "../data/2d_masks/"
    batch_size = 2
    tr_paths, v_paths = get_train_val_paths(image_folder, masks_folder)
    train_gen = mini_batch_data(tr_paths["train_imgs"], tr_paths["train_mask"], batch_size=batch_size)
    no_samples = len(tr_paths["train_imgs"])
    unet = UnetModel()
    unet.train(data_gen=train_gen, n_epochs=5, n_samples=no_samples)
    # import numpy as np
    # x_train = np.random.normal(size=(2, 512, 512, 1))
    # y_train = np.random.normal(size=(2, 512, 512, 1))
    #
    # X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 512, 512, 1], name="x")
    # y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 512, 512, 1], name="y")
    # unet = UnetModel()
    # output = unet.build_model(X)
    #
    # init = tf.compat.v1.global_variables_initializer()
    # with tf.compat.v1.Session() as sess:
    #     init.run()
    #     print(sess.run(output, feed_dict={X: x_train}))
