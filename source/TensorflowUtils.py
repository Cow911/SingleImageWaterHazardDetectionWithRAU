import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io
from tensorflow.contrib import rnn

from tensorflow.python.ops import array_ops

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha1=1, alpha2=1, gamma=2):
    
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha1 * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - alpha2 * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_mean(per_entry_cross_ent)

def save_image(image, save_dir, name, mean=None):
    if mean:
        image = unprocess_image(image, mean)
    misc.imsave(os.path.join(save_dir, name + ".png"), image)


def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var

def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def get_tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def conv3d_basic(x, W, bias):
    conv = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def conv3d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] *= 2
        output_shape[4] = W.get_shape().as_list()[3]
    conv = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def max_pool3D_2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding="SAME")

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def max_pool_mxn(x, m, n):
    return tf.nn.max_pool(x, ksize=[1, m, n, 1], strides=[1, m, n, 1], padding="SAME")

def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2, stride_y = 2):
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride_y, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def rnn_lstm(x, W, b, num_hidden):
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    output, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    rnn_res = x * W + b
    return rnn_res

def leaky_relu(x, alpha=0.0, name=""):
    return tf.maximum(alpha * x, x, name)

def max_pool_kxk(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def RA_unit(x, h, w, n):
    x_1 = tf.nn.avg_pool(x, ksize=[1, h/n, 2, 1], strides=[1, h/n, 2, 1], padding="SAME")
    print("h and n :",h,n)
    x_t = tf.zeros([1, h, w, 0], tf.float32)
    x_2 = tf.zeros([1, h, w, 0], tf.float32)
    x_t_small = tf.zeros([1, x_1.shape[1].value, w/2, 0], tf.float32)
    for k in range(n):
	x_t_1 = tf.slice(x_1, [0,k,0,0], [1,1,w/2,x.shape[3].value])
	x_t_2 = tf.image.resize_images(x_t_1, [h,w], 1)
        x_2 = tf.concat([x_2, x_t_2], axis=3)
	x_t_3 = tf.abs(x - x_t_2)
    	x_t = tf.concat([x_t, x_t_3], axis=3)
    x_out = tf.concat([x, x_t], axis=3)
    return x_out , x_2

def local_global_salience_unit(x, h, w):
    l_1 = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    l_2 = tf.image.resize_images(l_1, [h, w], 1)
    l_3 = x - l_2
    g_1 = tf.nn.avg_pool(x, ksize=[1, h, w, 1], strides=[1, h, w, 1], padding="SAME")
    g_2 = tf.image.resize_images(g_1, [h, w], 1)
    g_3 = x - g_2
    gl_fea = tf.concat([l_1, g_1], axis=3)

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed


def process_image(image, mean_pixel):
    return image - mean_pixel


def unprocess_image(image, mean_pixel):
    return image + mean_pixel


def bottleneck_unit(x, out_chan1, out_chan2, down_stride=False, up_stride=False, name=None):

    def conv_transpose(tensor, out_channel, shape, strides, name=None):
        out_shape = tensor.get_shape().as_list()
        in_channel = out_shape[-1]
        kernel = weight_variable([shape, shape, out_channel, in_channel], name=name)
        shape[-1] = out_channel
        return tf.nn.conv2d_transpose(x, kernel, output_shape=out_shape, strides=[1, strides, strides, 1],
                                      padding='SAME', name='conv_transpose')

    def conv(tensor, out_chans, shape, strides, name=None):
        in_channel = tensor.get_shape().as_list()[-1]
        kernel = weight_variable([shape, shape, in_channel, out_chans], name=name)
        return tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME', name='conv')

    def bn(tensor, name=None):
        return tf.nn.lrn(tensor, depth_radius=5, bias=2, alpha=1e-4, beta=0.75, name=name)

    in_chans = x.get_shape().as_list()[3]

    if down_stride or up_stride:
        first_stride = 2
    else:
        first_stride = 1

    with tf.variable_scope('res%s' % name):
        if in_chans == out_chan2:
            b1 = x
        else:
            with tf.variable_scope('branch1'):
                if up_stride:
                    b1 = conv_transpose(x, out_chans=out_chan2, shape=1, strides=first_stride,
                                        name='res%s_branch1' % name)
                else:
                    b1 = conv(x, out_chans=out_chan2, shape=1, strides=first_stride, name='res%s_branch1' % name)
                b1 = bn(b1, 'bn%s_branch1' % name, 'scale%s_branch1' % name)

        with tf.variable_scope('branch2a'):
            if up_stride:
                b2 = conv_transpose(x, out_chans=out_chan1, shape=1, strides=first_stride, name='res%s_branch2a' % name)
            else:
                b2 = conv(x, out_chans=out_chan1, shape=1, strides=first_stride, name='res%s_branch2a' % name)
            b2 = bn(b2, 'bn%s_branch2a' % name, 'scale%s_branch2a' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2b'):
            b2 = conv(b2, out_chans=out_chan1, shape=3, strides=1, name='res%s_branch2b' % name)
            b2 = bn(b2, 'bn%s_branch2b' % name, 'scale%s_branch2b' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2c'):
            b2 = conv(b2, out_chans=out_chan2, shape=1, strides=1, name='res%s_branch2c' % name)
            b2 = bn(b2, 'bn%s_branch2c' % name, 'scale%s_branch2c' % name)

        x = b1 + b2
        return tf.nn.relu(x, name='relu')


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)
