# Mostly follows:
# https://arxiv.org/pdf/1603.08155.pdf
# http://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf

import config
from keras.layers import Add, BatchNormalization, Conv2D, Cropping2D, Input, Lambda
from keras.models import Model
import tensorflow as tf

def residual_transform_block(input_tensor, num_filters):
    residual = Conv2D(
        filters = num_filters,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = 'relu',
    )(input_tensor)
    residual = BatchNormalization()(residual)
    residual = Conv2D(
        filters = num_filters,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = 'relu',
    )(residual)
    residual = BatchNormalization()(residual)

    cropped_input = Cropping2D(
        ((2, 2), (2, 2))
    )(input_tensor)
    return Add()([residual, cropped_input])

def build():
    input_tensor = Input(shape = config.DIMS)

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
    ))(input_tensor)

    # Begin squishing the image down! First just a stride 1 convolution.
    squished_image = Conv2D(
        filters = 32,
        kernel_size = (9, 9),
        strides = (1, 1),
        padding = 'SAME',
        activation = 'relu',
    )(padded_input_tensor)
    squished_image = BatchNormalization()(squished_image)
    squished_image = Conv2D(
        filters = 64,
        kernel_size = (3, 3),
        strides = (2, 2),
        padding = 'SAME',
        activation = 'relu',
    )(squished_image)
    squished_image = BatchNormalization()(squished_image)
    squished_image = Conv2D(
        filters = 128,
        kernel_size = (3, 3),
        strides = (2, 2),
        padding = 'SAME',
        activation = 'relu',
    )(squished_image)
    squished_image = BatchNormalization()(squished_image)

    # Squishing complete! Begin the glorious transformation!
    transformed_image = residual_transform_block(
        squished_image, num_filters = 128
    )
    transformed_image = residual_transform_block(
        transformed_image, num_filters = 128
    )
    transformed_image = residual_transform_block(
        transformed_image, num_filters = 128
    )
    transformed_image = residual_transform_block(
        transformed_image, num_filters = 128
    )
    transformed_image = residual_transform_block(
        transformed_image, num_filters = 128
    )

    # Begin the upscaling
    # Bilinear resize recommended by https://distill.pub/2016/deconv-checkerboard/
    upscaled = Lambda(lambda transformed_image: tf.image.resize_bilinear(
        transformed_image,
        (config.DIMS[0] // 2, config.DIMS[1] // 2)
    ))(transformed_image)
    upscaled = Conv2D(
        filters = 64,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = 'relu',
        padding = 'SAME',
    )(upscaled)
    upscaled = BatchNormalization()(upscaled)
    # Round 2 of upscaling!
    upscaled = Lambda(lambda upscaled: tf.image.resize_bilinear(
        upscaled,
        (config.DIMS[0], config.DIMS[1]),
    ))(upscaled)
    upscaled = Conv2D(
        filters = 64,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = 'relu',
        padding = 'SAME',
    )(upscaled)
    upscaled = BatchNormalization()(upscaled)
    # One big convolution to finish things up!
    upscaled = Conv2D(
        filters = 3,
        kernel_size = (9, 9),
        strides = (1, 1),
        activation = 'relu',
        padding = 'SAME',
    )(upscaled)

    return Model(input_tensor, upscaled)
