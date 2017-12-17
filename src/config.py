CONTENT_LAYER_NAME = "block4_conv2"
DIMS = (768, 1024, 3)

STYLE_PHOTO_PATH = 'images/starry-night1024.jpeg'
CONTENT_PHOTO_PATH = 'images/tubingen1024.jpeg'
#INPUT_IMAGE_PATH = 'outputs/new_image_1000.png'
INPUT_IMAGE_PATH = None
OUTPUT_IMAGE_PATH = 'outputs/new_image_b{:04}.png'
INITIAL_EPOCH = 0

LEARNING_RATE = 10.0
LOSS_WEIGHTS = [5, *([1/5 * 5e2] * 5)]
