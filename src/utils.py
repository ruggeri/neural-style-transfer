import config
from PIL import Image
import numpy as np

RGB_MEANS = np.array([123.68, 116.779, 103.939], dtype = np.float64)

def open_image(fname):
    img_data = np.array(
        Image.open(fname), dtype = np.float64
    )
    img_data = img_data - RGB_MEANS
    img_data[:, :, 2], img_data[:, :, 0] = (
        np.copy(img_data[:, :, 0]), np.copy(img_data[:, :, 2])
    )

    return img_data

def save_image(idx, new_image):
    new_image = new_image.reshape(config.DIMS)
    new_image[:, :, 2], new_image[:, :, 0] = (
        np.copy(new_image[:, :, 0]), np.copy(new_image[:, :, 2])
    )
    new_image = new_image + RGB_MEANS

    new_image = np.clip(new_image, 0, 255)
    new_image = new_image.astype(np.uint8)

    new_image = Image.fromarray(new_image, 'RGB')
    new_image.save(f'outputs/new_image_{idx:04}.png')
