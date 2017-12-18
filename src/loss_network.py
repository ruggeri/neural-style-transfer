import config
from keras.applications.vgg16 import VGG16
import keras.backend as K
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

def build(input_tensor):
    vgg = VGG16(
        include_top = False,
        weights = 'imagenet',
        input_tensor = input_tensor,
        pooling = 'avg',
    )

    # Freeze all layers of VGG16 except the input layer.
    for idx, layer in enumerate(vgg.layers):
        if idx > 0:
            vgg.layers[idx].trainable = False
        else:
            vgg.layers[0].trainable = True
            vgg.layers[0].trainable_weights.append(
                input_tensor
            )

    # Use the features of one of the final convolution layers.
    content_featurization_tensor = vgg.get_layer(
        config.CONTENT_LAYER_NAME
    ).output

    # == Style Featurization Tensors ==
    def style_matrix_tensor(conv_output_tensor):
        num_pixels = (
            conv_output_tensor.shape[1].value
            * conv_output_tensor.shape[2].value
        )
        num_channels = conv_output_tensor.shape[3].value
        num_elements_per_image = (num_channels * num_pixels)

        # Flatten from 4d to 3d: (batch_size, num_layers, width*height)
        channel_first_output = K.permute_dimensions(
            conv_output_tensor, (0, 3, 1, 2)
        )
        flattened_filter_vectors = K.reshape(
            channel_first_output,
            (-1, channel_first_output.shape[1], num_pixels)
        )
        transposed_flattened_filter_vectors = K.permute_dimensions(
            flattened_filter_vectors,
            (0, 2, 1)
        )

        style_matrix = K.batch_dot(
            flattened_filter_vectors,
            transposed_flattened_filter_vectors
        ) / num_elements_per_image
        print(style_matrix.shape)

        return style_matrix

    conv_layers = [
        vgg.get_layer(f'block{idx}_conv1') for idx in range(1, 6)
    ]
    style_matrix_tensors = [
        Lambda(style_matrix_tensor)(conv_layer.output)
        for conv_layer in conv_layers
    ]

    # == Losses ==
    def content_loss(y_true, y_pred):
        return K.mean(
            0.5 * K.square(y_true - y_pred)
        )

    def style_loss(y_true, y_pred):
        num_elements = y_pred.shape.num_elements()
        return K.mean(
            0.5 * K.square(y_true - y_pred)
        )

    model = Model(
        # We need to give this bogus input because otherwise Keras will
        # think the graph is "disconnected" and doesn't work back to an
        # input tensor. We'll never actually provide this input.
        [input_tensor],
        [content_featurization_tensor, *style_matrix_tensors],
    )

    model.compile(
        loss = [content_loss, *([style_loss] * 5)],
        loss_weights = config.LOSS_WEIGHTS,
        optimizer = Adam(lr = config.LEARNING_RATE),
    )

    return {
        "model": model,
        "content_featurization_tensor": content_featurization_tensor,
        "style_matrix_tensors": style_matrix_tensors
    }
