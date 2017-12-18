# This is the slow optimization approach to finding a stylized image.

import config
import loss_network
import utils
import keras.backend as K
from keras.callbacks import LambdaCallback
from keras.layers import Dense, Input, Reshape
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

# The weights in this dense matrix will be our image. This is a hack!
input_tensor = Input(shape = (1,))
image_layer = Dense(
    config.DIMS[0] * config.DIMS[1] * config.DIMS[2],
    activation = 'linear',
    use_bias = False,
)
image_tensor = Reshape(config.DIMS)(image_layer(input_tensor))

model = loss_network.build(input_shape = config.DIMS)
outputs = model(image_tensor)

model = Model(input_tensor, outputs)
model.compile(
    loss = 'mse',
    loss_weights = config.LOSS_WEIGHTS,
    optimizer = Adam(lr = config.LEARNING_RATE),
)

print(model.summary())

# == Read Content Image; Produce Featurization ==
print("Featurizing content photo!")
content_target_image = utils.open_image(config.CONTENT_PHOTO_PATH)
K.set_value(
    image_layer.weights[0],
    np.expand_dims(content_target_image.flatten(), axis = 0)
)
content_target_featurization, *_ = model.predict(np.ones((1, 1)))

# == Read Style Image; Produce Style Featurization ==
print("Featurizing style photo!")
style_target_image = utils.open_image(config.STYLE_PHOTO_PATH)
K.set_value(
    image_layer.weights[0],
    np.expand_dims(style_target_image.flatten(), axis = 0)
)
_, *style_target_featurizations = model.predict(np.ones((1, 1)))

def save_image_callback(epoch_idx, logs):
    if epoch_idx % 1 == 0:
        utils.save_image(
            epoch_idx, K.eval(image_layer.weights[0]).reshape(config.DIMS)
        )

# Start from content image.
# TODO: Why this random scaling of the normal noise?
if config.INPUT_IMAGE_PATH is None:
    initial_input_image = np.random.normal(size = config.DIMS) * 0.256
else:
    initial_input_image = utils.open_image(config.INPUT_IMAGE_PATH)
utils.save_image(0, initial_input_image)
K.set_value(
    image_layer.weights[0],
    np.expand_dims(initial_input_image.flatten(), axis = 0)
)

model.fit(
    # This model only has native tensor inputs.
    np.ones((1, 1)),
    [content_target_featurization, *style_target_featurizations],
    epochs = 10000,
    callbacks = [LambdaCallback(on_epoch_end = save_image_callback)],
    initial_epoch = config.INITIAL_EPOCH,
)
