CONTENT_LAYER_NAME = "block4_conv2"
DIMS = (256, 256, 3)

STYLE_PHOTO_PATH = 'images/starry-night256.jpeg'
CONTENT_PHOTO_PATH = 'images/tubingen256.jpeg'
#INPUT_IMAGE_PATH = 'outputs/new_image_1000.png'
INPUT_IMAGE_PATH = None
OUTPUT_IMAGE_PATH = 'outputs/new_image_b{:04}.png'
INITIAL_EPOCH = 0

TRAINING_INPUT_DIRECTORY = '/ebs/imagenet/ILSVRC2014_DET_train'

LEARNING_RATE = 0.01
LOSS_WEIGHTS = [5, *([1/5 * 5e2] * 5)]
BATCH_SIZE = 1
