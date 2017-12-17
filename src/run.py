import config
import network
import utils
import keras.backend as K
from keras.callbacks import LambdaCallback
import numpy as np

# == Read Content Image; Produce Featurization ==
print("Featurizing content photo!")
content_target_image = utils.open_image(config.CONTENT_PHOTO_PATH)
K.set_value(
    network.input_tensor,
    np.expand_dims(content_target_image, axis = 0)
)
content_target_featurization = K.eval(
    network.content_featurization_tensor
)

# == Read Style Image; Produce Style Featurization ==
print("Featurizing style photo!")
style_target_image = utils.open_image(config.STYLE_PHOTO_PATH)
K.set_value(
    network.input_tensor,
    np.expand_dims(style_target_image, axis = 0)
)
style_target_featurizations = [
    K.eval(style_matrix_tensor)
    for style_matrix_tensor in network.style_matrix_tensors
]

def save_image_callback(epoch_idx, logs):
    if epoch_idx % 100 == 0:
        utils.save_image(
            epoch_idx, K.eval(network.input_tensor)
        )

# Start from content image.
initial_input_image = np.random.normal(size = config.DIMS) * 0.256
utils.save_image(0, initial_input_image)
K.set_value(
    network.input_tensor,
    np.expand_dims(initial_input_image, axis = 0)
)
network.model.fit(
    [],
    [content_target_featurization, *style_target_featurizations],
    epochs = 10000,
    callbacks = [LambdaCallback(on_epoch_end = save_image_callback)]
)
