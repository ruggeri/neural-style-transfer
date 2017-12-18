# This is the hopefully faster approach to training a generator
# network.

import config
import generation_network
import keras.backend as K
from keras.layers import Input
import numpy as np
import utils

# == Encode the style image! ==
style_target_image = utils.open_image(config.STYLE_PHOTO_PATH)

input_tensor = Input(shape = config.DIMS)
encoding_model = loss_network.build(input_shape = config.DIMS)
_, *style_target_featurizations = encoding_model.predict(
    # Expand dims so it's a batch of one!
    np.expand_dims(style_target_image, axis = 0)
)

# == Encode the source image! ==
content_target_image = utils.open_image(config.CONTENT_PHOTO_PATH)
content_target_featurization, *_ = encoding_model.predict(
    # Expand dims so it's a batch of one!
    np.expand_dims(content_target_image, axis = 0)
)
# Clean up!
K.clear_session()

input_tensor = Input(config.DIMS)
generation_model = generation_network.build()
# TODO: here is where I should load weights!

stylized_image = generation_model.predict(
    np.expand_dims(content_target_image, axis = 0)
)
utils.save_image(0, stylized_image)
