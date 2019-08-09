# _*_ coding: utf-8 _*_
# Author: Jielong
# Creation Time: 25/07/2019 09:16
import tensorflow as tf


def weight_init(w_shape, std, name="weights"):
    """
    Initialize weights for kernel/filters
    :param w_shape: [kernel_size, kernel_size, channels of feature map/inputs, output_feature_map_channels]
    :param std: standard deviation set for weights initializer
    :param name: the weight name
    :return: weights using tf.Variable
    """
    with tf.name_scope(name):
        # tf.truncated_normal is deprecated
        weights = tf.random.truncated_normal(shape=w_shape, stddev=std)
        return tf.Variable(initial_value=weights, trainable=True, name=name)


def bias_init(value, shape, name=None):
    """
    create bias
    :param value: the value used for initializing bias
    :param shape: the 1D tensor array with a shape [output_feature_channels]
    :param name: the name for bias
    :return: bias
    """
    return tf.Variable(tf.constant(dtype=tf.float32, value=value, shape=shape), name=name)


def conv2d(X, W, b, rate):
    """
    convolution operation
    :param X: input with shape of (batch_size, height, width, channels)
    :param W: weights with shape of (kernel_size, kernel_size, input_feature_channels, output_feature_channels)
    :param b: bias with shape of [output_feature_channels]
    :param rate: the rate to keep for dropout operation
    :return: dropout result
    """
    with tf.name_scope("Conv2D"):
        conv = tf.nn.conv2d(input=X, filter=W, strides=1, padding="SAME")
        conv = tf.nn.bias_add(value=conv, bias=b)
        return tf.nn.dropout(x=conv, rate=rate)


def max_pool(X, name=None):
    """
    Maxpooling operation
    :param X: input features
    :param name: the name for maxpooling op
    :return: pooling result
    """
    return tf.nn.max_pool2d(input=X, ksize=2, strides=2, padding="VALID", name=name)


def deconv2d(X, W, strides=2):
    """
    Deconvolution operation
    :param X: input features
    :param W: the weights with shape of [kernel_size, kernel_size, input_feature_maps, output_feature_maps]
    :param strides: the times to transform back the image
    :return: deconvolution result
    """
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(X)
        output_shape = tf.stack([x_shape[0], x_shape[1] * strides, x_shape[2] * strides, x_shape[3] // 2])
        deconv = tf.nn.conv2d_transpose(value=X, filter=W, output_shape=output_shape,
                                        strides=strides, padding="SAME", name="deconvolution")
        # deconv = tf.nn.bias_add(value=deconv, bias=b)
        return deconv


def crop_and_copy(x_prev, x):
    """
    Concatenation and Cropping operation, concatenate the feature map results from downsampling path
    with upsampling path from corresponding layer
    :param x_prev: feature maps from downsampling path
    :param x: feature maps from upsampling path
    :return: concatenation result
    """
    with tf.name_scope("crop_and_concatenation"):
        x_prev_shape = tf.shape(x_prev)
        # print("The original shape: ", x_prev.shape)
        x_shape = tf.shape(x)
        # print("The upsampling shape: ", x.shape)
        offset = [0, (x_prev_shape[1] - x_shape[1]) // 2, (x_prev_shape[2] - x_shape[2]) // 2, 0]
        x_prev_cropped = tf.slice(x_prev, offset, size=(-1, x_shape[1], x_shape[2], -1))
        return tf.concat([x_prev_cropped, x], axis=3)


def piexel_wise_softmax(X):
    with tf.name_scope("pixel_wise_softmax"):
        pass


if __name__ == "__main__":
    # pass
    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)
    # x_train = np.random.normal(size=(2, 512, 512, 1))
    # y_train = np.random.normal(size=(2, 512, 224, 1))

    X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 512, 512, 1], name="x")
    y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 512, 512, 1], name="y")

    # Extraction Path
    conv_ops = 2
    model_depth = 5
    k_size, pool_size, feat_maps_root = 3, 2, 16
    convs_rslt = []

    for depth in range(model_depth):
        print("the down level is: ", depth)
        feature_maps = 2 ** depth * feat_maps_root
        print("Feature maps = ", feature_maps)
        stddev = tf.sqrt(2 / (k_size ** 2 * feature_maps))

        convs_temp = []
        conv = convs_rslt[depth - 1][1] if convs_rslt else X
        for conv_op in range(conv_ops):
            if depth == 0:
                input_feat_channels = tf.shape(X)[3]
            else:
                input_feat_channels = tf.shape(convs_rslt[depth - 1][0][1])[3]
            W = weight_init(w_shape=(k_size, k_size, input_feat_channels, feature_maps), std=stddev)
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

        if depth == model_depth - 1:
            rslt_for_deconv = convs_temp[1]
            print("After downsampling, the shape is: ", rslt_for_deconv.shape)
            convs_rslt.append(convs_temp)
        else:
            pool = max_pool(convs_temp[1])
            # X = pool
            print("After max pooling: ", pool.shape)
            print("\n")
            convs_rslt.append((convs_temp, pool))   # conv_rslt[0][1], conv_rslt[1][1], conv_rslt[2][1], conv_rslt[3][1]

    # Expansive Path
    print("\n")
    deconv_rslt = []
    step = -1
    for depth in range(model_depth - 2, -1, -1):
        print("The up level is: ", depth)
        feature_maps = 2 ** (depth + 1) * feat_maps_root
        print("the up feature maps are: ", feature_maps)
        stddev = tf.sqrt(2 / (k_size ** 2 * feature_maps))
        # conv = X
        conv = convs_rslt[-1][1] if depth == model_depth - 2 else deconv_rslt[step][1]
        W_d = weight_init(w_shape=[pool_size, pool_size, feature_maps // 2, feature_maps], std=stddev)
        # print(W_d.shape)
        b_d = bias_init(value=0.1, shape=[feature_maps // 2], name="up_bias_{0}".format(depth))
        # print(b_d.shape)
        deconv = deconv2d(conv, W=W_d, strides=pool_size)
        concat_deconv_conv = crop_and_copy(convs_rslt[depth][0][1], deconv)
        print("After deconv: ", deconv.shape)
        print("concat result: ", concat_deconv_conv.shape)

        # X = concat_deconv_conv
        convs_temp = []
        for conv_op in range(conv_ops):
            b = bias_init(value=0.1, shape=[feature_maps // 2], name="up_bias_{0}_{1}".format(depth, conv_op))
            if conv_op == 0:
                W = weight_init(w_shape=[k_size, k_size, feature_maps, feature_maps // 2], std=stddev)
                conv = conv2d(X=concat_deconv_conv, W=W, b=b, rate=0.2)
                conv = tf.nn.elu(features=conv)
                print("Shape of data after upsamling and convolution: ", conv.shape)
            else:
                W = weight_init(w_shape=[k_size, k_size, feature_maps // 2, feature_maps // 2], std=stddev)
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
        stddev = tf.sqrt(2 / (k_size ** 2 * feat_maps_root))
        W = weight_init(w_shape=(1, 1, feat_maps_root, 1), std=stddev)
        b = bias_init(value=0.1, shape=[1], name="final_out_bias")
        conv = conv2d(X=deconv_rslt[-1][1], W=W, b=b, rate=0.1)
        output = tf.nn.sigmoid(conv)
        print("final output shape", output.shape)

    with tf.name_scope("training_op"):
        loss = tf.compat.v1.losses.sigmoid_cross_entropy(y, output)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
        training_op = optimizer.minimize(loss)

    n_epochs = 5
    image_folder = "../data/2d_images/"
    masks_folder = "../data/2d_masks/"
    batch_size = 2

    from utils import get_train_val_paths, mini_batch_data

    tr_paths, v_paths = get_train_val_paths(image_folder, masks_folder)
    train_gen = mini_batch_data(tr_paths["train_imgs"], tr_paths["train_mask"], batch_size=batch_size)
    x_batch, y_batch = next(train_gen)
    print(x_batch.shape)
    print("The shape of y_true: ", y_batch.shape)
    # max_val = tf.math.maximum(X)
    # min_val = tf.math.minimum(X)
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        init.run()
        # print(sess.run(output, feed_dict={X: x_batch}))
        print(output.eval(feed_dict={X: x_batch}))
        # loss_val, _ = sess.run([loss, training_op], feed_dict={X: x_batch, y: y_batch})
        # print(loss_val)
