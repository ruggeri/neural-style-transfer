CONTENT_LAYER_NAME = "block4_conv2"
DIMS = (768, 1024, 3)

STYLE_PHOTO_PATH = 'images/starry-night1024.jpeg'
CONTENT_PHOTO_PATH = 'images/tubingen1024.jpeg'

LEARNING_RATE = 1.0
LOSS_WEIGHTS = [1e-3, *([1/5] * 5)]
