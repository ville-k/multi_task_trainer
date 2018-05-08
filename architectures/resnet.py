# Copyright (C) 2018, Ville Kallioniemi <ville.kallioniemi@gmail.com>

import numpy as np
import tensorflow as tf


def convolution(inputs, input_filters, strides=[1, 1, 1, 1]):
    fan_in = input_filters[0] * input_filters[1] * input_filters[2]
    fan_out = input_filters[0] * input_filters[1] * input_filters[3]

    W = tf.Variable(dtype=tf.float32,
                    initial_value=tf.truncated_normal(
                        input_filters, stddev=np.math.sqrt(1.0 / (fan_in + fan_out)))
                    )
    tf.summary.histogram('weights', W)
    b = tf.Variable(dtype=tf.float32,
                    initial_value=tf.constant(0.1, shape=[input_filters[3]])
                    )
    convolved = tf.nn.conv2d(inputs, W, strides=strides, padding='SAME')
    return convolved + b


def linear(inputs, input_count, output_count):
    W = tf.Variable(dtype=tf.float32,
                    initial_value=tf.truncated_normal((input_count, output_count),
                                                      stddev=np.math.sqrt(1 / (input_count + output_count)))
                    )
    b = tf.Variable(dtype=tf.float32,
                    initial_value=tf.constant(0.1, shape=[output_count])
                    )
    return tf.matmul(inputs, W) + b


def contract(inputs, shape, stride):
    input_shape = inputs.get_shape().as_list()

    row_stride = 1 if input_shape[1] <= shape[1] else stride
    column_stride = 1 if input_shape[2] <= shape[2] else stride
    channel_stride = 1 if input_shape[3] <= shape[3] else stride

    contracted = inputs
    if row_stride != 1:
        contracted = contracted[:, 0:input_shape[1]:row_stride, :, :]
    if column_stride != 1:
        contracted = contracted[:, :, 0:input_shape[2]:column_stride, :]
    if channel_stride != 1:
        contracted = contracted[:, :, :, 0:input_shape[3]:channel_stride]

    return contracted


def expand(inputs, shape):
    input_shape = inputs.get_shape().as_list()

    row_padding = 0 if input_shape[1] >= shape[1] else shape[1] - \
        input_shape[1]
    column_padding = 0 if input_shape[2] >= shape[2] else shape[2] - \
        input_shape[2]
    channel_padding = 0 if input_shape[3] >= shape[3] else shape[3] - \
        input_shape[3]

    if row_padding > 0 or column_padding > 0 or channel_padding > 0:
        return tf.pad(inputs, [[0, 0], [0, row_padding], [0, column_padding], [0, channel_padding]], name='zero_pad')
    else:
        return inputs


def contract_expand_residual(inputs, shape, stride=2):
    with tf.name_scope("contract"):
        residual = contract(inputs, shape, stride)
    with tf.name_scope("expand"):
        return expand(residual, shape)


def resnet_block(inputs, shape, stride=1):
    input_filter = shape
    input_stride = [1, stride, stride, 1]

    convolved = convolution(inputs, input_filter, input_stride)
    mean, variance = tf.nn.moments(convolved, axes=[0, 1, 2], keep_dims=False)
    normalized = tf.nn.batch_normalization(
        convolved, mean, variance, offset=None, scale=None, variance_epsilon=0.00001)
    activated = tf.nn.relu(normalized)

    output_filter = [shape[0], shape[1], shape[3], shape[3]]
    convolved = convolution(activated, output_filter)

    mean, variance = tf.nn.moments(convolved, axes=[0, 1, 2], keep_dims=False)
    normalized = tf.nn.batch_normalization(
        convolved, mean, variance, offset=None, scale=None, variance_epsilon=0.00001)

    # Objective is to reshape the inputs to a shape that can be summed with the convolved values
    # in:  [?, 28, 28,  64]
    # convolve with stride=2 and increase filters to 128
    # convolved: [?, 14, 14, 128]
    # => residual must match convolved dims
    residual = contract_expand_residual(
        normalized, convolved.get_shape().as_list())

    return tf.nn.relu(tf.add(convolved, residual))


def model(out):
    """Create a model.

    Args:
        out (tf.Tensor): Outpus from previous (input) layer.
    Returns:
        (tf.Tensor): outputs from this model.
    """
    input_channels = out.get_shape().as_list()[-1]

    with tf.name_scope("ResNet_1"):
        out = resnet_block(out, [3, 3, input_channels, 64])

    with tf.name_scope("ResNet_2"):
        out = resnet_block(out, [3, 3, 64, 128], stride=2)

    return out
