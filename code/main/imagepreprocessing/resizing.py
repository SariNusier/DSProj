# Helper module for resizing images
from skimage import io
from skimage import transform
import numpy as np


def resize(img):
    return transform.pyramid_expand(img)


def read_image(path):
    return io.imread(path)


def print_image(img):
    io.imshow(img)
