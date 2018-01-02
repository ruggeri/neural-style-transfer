import config256 as config
import generation_network
from keras.callbacks import LambdaCallback, ReduceLROnPlateau
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
#from keras.preprocessing.image import ImageDataGenerator
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
style_target_content, *style_target_featurizations = encoding_model.predict(
    # Expand dims so it's a batch of one!
    np.expand_dims(style_target_image, axis = 0)
)

# Setup combined generation and loss network.
generation_input_tensor = Input(config.DIMS)
generation_model = generation_network.build(config.DIMS)
#generation_model.load_weights(config.MODEL_PATH)
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

image_regexp = re.compile('.*\.jpg')
def image_paths():
    for path in os.listdir(config.TRAINING_INPUT_DIRECTORY):
        path = os.path.join(config.TRAINING_INPUT_DIRECTORY, path)
        if not os.path.isfile(path): continue
        if not image_regexp.match(path): continue
        yield path

from PIL import Image
def images(image_paths):
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.resize(config.DIMS[0:2])
        image_data = np.array(image, dtype = np.float64)
        # Handle grayscale images
        if len(image_data.shape) == 2:
            image_data = image_data[:, :, np.newaxis]
            image_data = np.repeat(image_data, repeats = 3, axis = 2)
        yield image_data

def image_batches(images):
    image_batch = []
    for image_data in images:
        image_batch.append(image_data)
        if len(image_batch) < config.BATCH_SIZE: continue

        image_batch = np.stack(image_batch)
        batch_content, *_ = encoding_model.predict(image_batch)

        # Do a final bit of preprocessing so that scale into generator
        # is not stupid.
        image_batch = image_batch / 255.0

        yield (
            image_batch,
            [batch_content, *batch_style_target_featurizations]
        )

        image_batch = []

def save_generation_model(epoch, logs):
    loss = logs['loss']
    fname = f"ckpts/generation_weights.styled2_E{epoch:04d}.L{loss:.3e}.hdf5"
    generation_model.save_weights(fname)

NUM_TRAINING_IMAGES = 118287

from queue import Queue
QUEUE_SIZE = 64
queue = Queue(maxsize = QUEUE_SIZE)

def worker():
    while True:
        for image_batch in image_batches(images(image_paths())):
            print(image_batch)
            queue.put(image_batch)

import threading
#t = threading.Thread(target = worker)
#t.start()

def consumer():
    while True:
        image_batch = queue.get()
        queue.task_done()
        yield image_batch

training_model.fit_generator(
    image_batches(images(image_paths())),
    epochs = 1000,
    initial_epoch = config.INITIAL_EPOCH,
    steps_per_epoch = (
        # I want to create 32x as many epochs so weights are saved
        # more often. Each epoch looks at 1/32nd of the data.
        NUM_TRAINING_IMAGES // (config.BATCH_SIZE * 32)
    ),
    callbacks = [
        LambdaCallback(on_epoch_end = save_generation_model),
        ReduceLROnPlateau(
            monitor = 'loss',
            factor = 0.5,
            patience = 1,
        ),
    ],
)
