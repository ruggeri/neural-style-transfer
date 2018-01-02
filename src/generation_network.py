# Mostly follows:
# https://arxiv.org/pdf/1603.08155.pdf
# http://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf

import keras.backend as K
from keras.initializers import Constant
from keras.layers import Activation, Add, BatchNormalization, Conv2D, Cropping2D, Input, Lambda
from keras.models import Model
import tensorflow as tf
import utils

def conv(input_tensor, num_filters, kernel_size, stride, use_relu, padding):
    output_tensor = Conv2D(
        filters = num_filters,
        kernel_size = (kernel_size, kernel_size),
        strides = (stride, stride),
        padding = padding,
    )(input_tensor)

    output_tensor = BatchNormalization()(output_tensor)
    if use_relu:
        output_tensor = Activation('relu')(output_tensor)

    return output_tensor

def residual_transform_block(input_tensor, num_filters, block_idx):
    residual = conv(
        input_tensor,
        num_filters,
        kernel_size = 3,
        stride = 1,
        padding = 'VALID',
        use_relu = True
    )
    residual = conv(
        residual,
        num_filters,
        kernel_size = 3,
        stride = 1,
        padding = 'VALID',
        use_relu = False
    )

    cropped_input = Cropping2D(
        ((2, 2), (2, 2)),
        name = f'resblock{block_idx}/crop'
    )(input_tensor)
    return Add(name = f'resblock{block_idx}/add')(
        [residual, cropped_input]
    )

def build(input_shape):
    input_tensor = Input(shape = input_shape)

    # We will do a number of valid padding convolutions which reduces
    # image dimensions. Therefore, we will extend the image so that we
    # get an output of the same size.
    padded_input_tensor = Lambda(lambda input_tensor: tf.pad(
        input_tensor, [
            (0, 0), # don't pad batch. duh.
            (40, 40),
            (40, 40),
            (0, 0), # don't pad channels. duh.
        ], 'REFLECT'
    ), name = 'padding')(input_tensor)

    # Begin squishing the image down! First just a stride 1 convolution.
    squished_image = conv(
        padded_input_tensor,
        num_filters = 32,
        kernel_size = 9,
        stride = 1,
        padding = 'SAME',
        use_relu = True,
    )
    squished_image = conv(
        squished_image,
        num_filters = 64,
        kernel_size = 3,
        stride = 2,
        padding = 'SAME',
        use_relu = True,
    )
    squished_image = conv(
        squished_image,
        num_filters = 128,
        kernel_size = 3,
        stride = 2,
        padding = 'SAME',
        use_relu = True,
    )

    # Squishing complete! Begin the glorious transformation!
    transformed_image = residual_transform_block(
        squished_image, num_filters = 128, block_idx = 1,
    )
    transformed_image = residual_transform_block(
        transformed_image, num_filters = 128, block_idx = 2,
    )
    transformed_image = residual_transform_block(
        transformed_image, num_filters = 128, block_idx = 3,
    )
    transformed_image = residual_transform_block(
        transformed_image, num_filters = 128, block_idx = 4,
    )
    transformed_image = residual_transform_block(
        transformed_image, num_filters = 128, block_idx = 5,
    )

    # Begin the upscaling
    # Bilinear resize recommended by https://distill.pub/2016/deconv-checkerboard/
    upscaled = Lambda(lambda transformed_image: tf.image.resize_bilinear(
        transformed_image,
        (input_shape[0] // 2, input_shape[1] // 2),
    ), name = 'upscale1/resize')(transformed_image)
    upscaled = Conv2D(
        filters = 64,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = 'relu',
        padding = 'SAME',
        name = 'upscale1/conv2d',
    )(upscaled)
    upscaled = BatchNormalization(name = 'upscale1/bn')(upscaled)
    # Round 2 of upscaling!
    upscaled = Lambda(lambda upscaled: tf.image.resize_bilinear(
        upscaled,
        (input_shape[0], input_shape[1]),
    ), name = 'upscale2/resize')(upscaled)
    upscaled = Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = 'relu',
        padding = 'SAME',
        name = 'upscale2/conv2d',
    )(upscaled)
    upscaled = BatchNormalization(name = 'upscale2/bn')(upscaled)

    # One big convolution to finish things up!
    upscaled = Conv2D(
        filters = 3,
        kernel_size = (9, 9),
        strides = (1, 1),
        activation = 'tanh',
        padding = 'SAME',
        name = 'final_convolution',
    )(upscaled)

    upscaled = Lambda(lambda ipt: (ipt+1)*127.5)(upscaled)

    return Model(input_tensor, upscaled)
