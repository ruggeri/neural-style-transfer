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

# Clean up!
K.clear_session()

input_tensor = Input(config.DIMS)
generation_model = generation_network.build()
generation_output = generation_model(input_tensor)
loss_model = loss_network.build(input_shape = config.DIMS)
training_outputs = loss_model(generation_output)

training_model = Model(input_tensor, training_outputs)
training_model.compile(
    loss = 'mse',
    loss_weights = config.LOSS_WEIGHTS,
    optimizer = Adam(config.LEARNING_RATE),
)

# TODO: this will just learn to produce a good stylization for a
# single input.
for training_image in training_images:
    training_model.fit(
        np.expand_dims(training_image, axis = 0),
        [content_target_featurization, *style_target_featurizations],
        epochs = 1,
        initial_epoch = config.INITIAL_EPOCH,
    )
