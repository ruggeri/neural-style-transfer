# This is the hopefully faster approach to training a generator
# network.

import config256 as config
import generation_network
import keras.backend as K
from keras.layers import Input
import numpy as np
import utils

# Load source image.
content_target_image = utils.open_image(config.CONTENT_PHOTO_PATH)

generation_model = generation_network.build(config.DIMS)
generation_model.load_weights(config.MODEL_PATH, by_name = True)

stylized_image = generation_model.predict(
    np.expand_dims(content_target_image, axis = 0)
)
utils.save_image(0, stylized_image, image_shape = config.DIMS)

