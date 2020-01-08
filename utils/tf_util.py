# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# ------------------------
#   FUNCTIONS
# ------------------------
def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """
        Helper to create a variable stored on CPU memory
        :param name: name of the variable
        :param shape: list of ints
        :param initializer: initializer for the variable
        :param use_fp16: boolean for float16 usage
        :return: variable tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """
        Helper to create an initialized variable with weight decay
        ***NOTE***: That the variable is initialized a truncated normal distribution
        and a weight decay is added only if one is specified
        :param name: name of the variable
        :param shape: list of ints
        :param stddev: standard deviation of a truncated gaussian
        :param wd: add L2Loss weight decay multiplied by this float. If None, weight decay is not added for this variable
        :param use_xavier: boolean for whether or not we use the xavier initializer
        :return: variable tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
