import config
import loss_network
import utils
import keras.backend as K
from keras.layers import Input
from keras.callbacks import LambdaCallback
import numpy as np

# Input is needed to instantiate a Keras tensor. It will create an
# InputLayer for you (Input is just a function).
input_tensor = Input(tensor = K.variable(np.zeros(
    shape = (1, *config.DIMS),
)))

network = loss_network.build(input_tensor)
model, content_featurization_tensor, style_matrix_tensors = (
    [network[key] for key in [
        "model", "content_featurization_tensor", "style_matrix_tensors"
    ]]
)

print(model.summary())

# == Read Content Image; Produce Featurization ==
print("Featurizing content photo!")
content_target_image = utils.open_image(config.CONTENT_PHOTO_PATH)
K.set_value(
    input_tensor,
    np.expand_dims(content_target_image, axis = 0)
)
content_target_featurization = K.eval(
    content_featurization_tensor
)

# == Read Style Image; Produce Style Featurization ==
print("Featurizing style photo!")
style_target_image = utils.open_image(config.STYLE_PHOTO_PATH)
K.set_value(
    input_tensor,
    np.expand_dims(style_target_image, axis = 0)
)
style_target_featurizations = [
    K.eval(style_matrix_tensor)
    for style_matrix_tensor in style_matrix_tensors
]

def save_image_callback(epoch_idx, logs):
    if epoch_idx % 1 == 0:
        utils.save_image(
            epoch_idx, K.eval(input_tensor)
        )

# Start from content image.
# TODO: Why this random scaling of the normal noise?
if config.INPUT_IMAGE_PATH is None:
    initial_input_image = np.random.normal(size = config.DIMS) * 0.256
else:
    initial_input_image = utils.open_image(config.INPUT_IMAGE_PATH)
utils.save_image(0, initial_input_image)
K.set_value(
    input_tensor,
    np.expand_dims(initial_input_image, axis = 0)
)
model.fit(
    # This model only has native tensor inputs.
    None,
    [content_target_featurization, *style_target_featurizations],
    epochs = 10000,
    callbacks = [LambdaCallback(on_epoch_end = save_image_callback)],
    initial_epoch = config.INITIAL_EPOCH,
)
