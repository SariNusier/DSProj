"""Helper module for resizing images"""

from skimage import io
from skimage import transform
from PIL import Image
import numpy as np

RESIZE_NONE = "Don't resize"
RESIZE_PW = "Padding and Sliding Window"
RESIZE_NEAREST = Image.NEAREST
RESIZE_BILINEAR = Image.BILINEAR
RESIZE_BICUBIC = Image.BICUBIC


def resize(images, width, height, mode=RESIZE_NONE):
    if mode == RESIZE_NONE:
        return images

    if mode == RESIZE_PW:
        results = []
        for img in images:
            resized_img = resize_pw(img, width, height)
            if len(resized_img) == 1:
                results.append(resized_img[0])
            else:
                for r in resized_img:
                    results.append(r)
        return np.array(results)

    results = []
    for img in images:
        resized_img = Image.fromarray(img)
        resized_img = resized_img.resize((width, height), mode)
        results.append(np.array(resized_img))

    return np.array(results)


def resize_pw(img, width, height):
    orig_height, orig_width = img.shape
    top = 0
    bot = 0
    left = 0
    right = 0

    if orig_width < width:
        left = (width - orig_width) / 2
        right = (width - orig_width) - left

    if orig_height < height:
        top = (height - orig_height) / 2
        bot = (height - orig_height) - top

    padded_img = np.lib.pad(img, ((top, bot), (left, right)), 'constant', constant_values=0)
    if padded_img.shape[0] > height or padded_img.shape[1] > width:
        return sliding_window(padded_img, width, height)
    padded_img = [padded_img]
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

        slide_down = min([height - bot_b, w_height])
        top_b = top_b + slide_down
        bot_b = bot_b + slide_down

    return results


def read_image(path):
    return io.imread(path)


def print_image(img):
    io.imshow(img)
