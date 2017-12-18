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
        # Flatten from 4d to 2d: (num_layers, width*height)
        channel_first_output = K.permute_dimensions(
            conv_output_tensor, (0, 3, 1, 2)
        )
        flattened_filter_vectors = K.reshape(
            channel_first_output,
            (channel_first_output.shape[1], -1)
        )

        style_matrix = K.dot(
            flattened_filter_vectors,
            K.transpose(flattened_filter_vectors)
        ) / flattened_filter_vectors.shape.num_elements()
        print(style_matrix.shape)
        print(style_matrix.shape.num_elements())

        # First silly dimension is for example idx in "batch."
        return K.expand_dims(style_matrix, axis = 0)

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
