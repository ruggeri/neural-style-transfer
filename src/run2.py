# This is the hopefully faster approach to training a generator
# network.

import config
import generation_network
import keras.backend as K
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import loss_network
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
generation_output = generation_model(input_tensor)
print(generation_output.shape)
loss_model = loss_network.build(input_shape = config.DIMS)
training_outputs = loss_model(generation_output)

training_model = Model(input_tensor, training_outputs)
training_model.compile(
    loss = 'mse',
    loss_weights = config.LOSS_WEIGHTS,
    optimizer = Adam(config.LEARNING_RATE),
)

# TODO: where will we get training data???

stylized_image = generation_model.predict(
    np.expand_dims(content_target_image, axis = 0)
)
utils.save_image(0, stylized_image)
