"""Helper module for resizing images"""

from skimage import io
from skimage import transform
from PIL import Image
import numpy as np


def resize(img, width, height, mode=0):
    resized_img = Image.fromarray(img)
    resized_img = resized_img.resize((width, height), mode)
    return np.array(resized_img.resize((width, height)))


def resize2(img, width, height):
    orig_height, orig_width = img.shape
    top = 0
    bot = 0
    left = 0
    right = 0

    if orig_width < width:
        left = (width - orig_width)/2
        right = (width - orig_width) - left

    if orig_height < height:
        top = (height - orig_height)/2
        bot = (height - orig_height) - top

    padded_img = np.lib.pad(img, ((top, bot), (left, right)), 'constant', constant_values=0)

    if padded_img.shape[0] > height or padded_img.shape[1] > width:
        # return sliding_window(padded_img, width, height)
        return np.array(Image.fromarray(padded_img).resize((28, 28)))

    return padded_img


def sliding_window(img, w_width, w_height):
    height, width = img.shape
    results = []
    right_b = w_width
    bot_b = w_height
    left_b = 0
    top_b = 0
    slide_down = 1

    while slide_down != 0:
        right_b = w_width
        left_b = 0
        slide_right = 1
        while slide_right != 0:
            results.append(img[top_b:bot_b, left_b:right_b])
            slide_right = min([width - right_b, w_width])
            left_b = left_b + slide_right
            right_b = right_b + slide_right
            print "Sliding right by: " + str(slide_right)

        slide_down = min([height - bot_b, w_height])
        top_b = top_b + slide_down
        bot_b = bot_b + slide_down

    return results



def read_image(path):
    return io.imread(path)


def print_image(img):
    io.imshow(img)
