import config256 as config
import generation_network
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import loss_network
import numpy as np
import os
import os.path
import re
import utils

# == Encode the style image! ==
style_target_image = utils.open_image(config.STYLE_PHOTO_PATH)

encoding_input_tensor = Input(shape = config.DIMS)
encoding_model = loss_network.build(input_shape = config.DIMS)
_, *style_target_featurizations = encoding_model.predict(
    # Expand dims so it's a batch of one!
    np.expand_dims(style_target_image, axis = 0)
)

# Setup combined generation and loss network.
generation_input_tensor = Input(config.DIMS)
generation_model = generation_network.build(config.DIMS)
generation_output = generation_model(generation_input_tensor)
training_outputs = encoding_model(generation_output)

training_model = Model(generation_input_tensor, training_outputs)
training_model.compile(
    loss = 'mse',
    loss_weights = config.LOSS_WEIGHTS,
    optimizer = Adam(config.LEARNING_RATE),
)

print("ENCODING MODEL")
print(encoding_model.summary())
print("GENERATOR MODEL")
print(generation_model.summary())
print("TRAINING MODEL")
print(training_model.summary())

# Just stack the style featurizations repeatedly; every batch input
# has the same style target.
batch_style_target_featurizations = [
    # s_matrix is shape (1, num_filters, num_filters)
    np.repeat(s_matrix, repeats = config.BATCH_SIZE, axis = 0)
    for s_matrix in style_target_featurizations
]

NUM_TRAINING_IMAGES = 0
for fname in os.listdir(config.TRAINING_INPUT_DIRECTORY):
    fname = os.path.join(config.TRAINING_INPUT_DIRECTORY, fname)
    if not re.match('^.*\.JPEG$', fname): continue
    NUM_TRAINING_IMAGES += 1

def training_generator():
    while True:
        training_images = []
        for fname in os.listdir(config.TRAINING_INPUT_DIRECTORY):
            fname = os.path.join(config.TRAINING_INPUT_DIRECTORY, fname)
            if not re.match('^.*\.JPEG$', fname): continue

            training_image = utils.open_image(fname)
            training_images.append(training_image)
            if len(training_images) < config.BATCH_SIZE: continue

            # Convert to numpy array.
            training_images_array = np.stack(training_images)
            training_images_content, *_ = encoding_model.predict(
                training_images_array
            )

            yield (
                training_images_array,
                [training_images_content, *batch_style_target_featurizations]
            )

            # Reset for the next batch.
            training_images = []

training_model.fit_generator(
    training_generator(),
    epochs = 1000,
    initial_epoch = config.INITIAL_EPOCH,
    steps_per_epoch = (
        # I want to create 32x as many epochs so weights are saved
        # more often. Each epoch looks at 1/32nd of the data.
        NUM_TRAINING_IMAGES // (config.BATCH_SIZE * 32)
    ),
    callbacks = [ModelCheckpoint(
        'ckpts/weights.e{epoch:04d}.l{loss:.3e}.hdf5',
    )]
)
