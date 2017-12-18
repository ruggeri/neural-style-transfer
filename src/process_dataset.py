import config
from keras.layers import Input
import loss_network
import numpy as np
import utils

input_tensor = Input(shape = config.DIMS)
encoding_model = loss_network.build(input_shape = config.DIMS)

for training_image in training_images:
    content_target_featurization, *_ = encoding_model.predict(
        # Expand dims so it's a batch of one!
        np.expand_dims(content_target_image, axis = 0)
    )
