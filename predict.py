# _*_ coding: utf-8 _*_
# Author: Jielong
# Creation Time: 10/08/2019 00:38
import tensorflow as tf
from utils import get_batch_data, get_imgs_masks


class ModelPredictionOp(object):

    def __init__(self):
        pass

    def restore_model(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    # x_test = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 512, 512, 1], name="x_input")
    # y_test = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 512, 512, 1], name="y_label")

    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph("./models/tf_model.ckpt.meta")
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint("./models/"))
    graph = tf.get_default_graph()
    # x_test = graph.compat.v1.get_tensor_by_name("x_input:0")
    x_test = graph.compat.v1.get_operation_by_name("x_input").outputs[0]
    y_test = tf.compat.v1.get_collection("network_architecture")[0]
    # op_to_restore = graph.get_tensor_by_name("final_output:0")
    # get_final_ouput_op = graph.get_tensor_by_name("Sigmoid:0")
    # for op in graph.get_operations():
    #     print(op)

    # image_folder = "./data/2d_images/"
    # masks_folder = "./data/2d_masks/"
    # images, labels = get_imgs_masks(image_folder, masks_folder)
    # batch_size = 4
    # no_samples = images.shape[0]
    # n_iterations = no_samples // batch_size
    # for i in range(n_iterations):
    #     x_batch, y_batch = get_batch_data(images, labels, iter_step=i, batch_size=4)
        # predictions = sess.run(get_final_ouput_op, feed_dict={x_test: x_batch, y_test: y_batch})
