import config
from PIL import Image
import numpy as np

RGB_MEANS = np.array([123.68, 116.779, 103.939], dtype = np.float32)
BGR_MEANS = np.array([103.939, 116.779, 123.68], dtype = np.float32)

# TODO: Not particularly efficient...
def batch_vgg_preprocess(batch_img_data):
    batch_img_data = np.copy(batch_img_data)
    for idx in range(batch_img_data.shape[0]):
        batch_img_data[idx, :, :, :] = vgg_preprocess(
            batch_img_data[idx, :, :, :]
        )

    return batch_img_data

def vgg_preprocess(img_data):
    img_data = np.copy(img_data)
    img_data -= RGB_MEANS

    img_data[:, :, 2], img_data[:, :, 0] = (
        np.copy(img_data[:, :, 0]), np.copy(img_data[:, :, 2])
    )

    return img_data

def open_image(fname, vggify = True):
    img_data = np.array(
        Image.open(fname), dtype = np.float32
    )

    # Used to do this when using ImageNet grayscale images.
    #if len(img_data.shape) == 2:
    #    # Grayscale image => Color
    #    img_data = np.repeat(
    #        img_data[:, :, np.newaxis], repeats = 3, axis = 2
    #    )

    if vggify:
        img_data = vgg_preprocess(img_data)

    return img_data

def save_image(idx, new_image, image_shape = config.DIMS, unvggify = True):
    # We make a copy to avoid accidentally swapping channels on the
    # user.
    new_image = np.copy(new_image).reshape(image_shape)
    if unvggify:
        new_image[:, :, 2], new_image[:, :, 0] = (
            np.copy(new_image[:, :, 0]), np.copy(new_image[:, :, 2])
        )
        new_image = new_image + RGB_MEANS

    new_image = np.clip(new_image, 0, 255)
    new_image = new_image.astype(np.uint8)

    new_image = Image.fromarray(new_image, 'RGB')
    new_image.save(config.OUTPUT_IMAGE_PATH.format(idx))
