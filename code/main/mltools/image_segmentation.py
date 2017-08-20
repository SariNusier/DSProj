"""
Module to implement object detection and image segmentation.
This should have multiple ways of segmenting the images and exporting the cropped objects.
"""
from imgpreproc import reading

a, b, c, d = reading.get_data_tt(5)
print a.shape
print b.shape
print c.shape
print d.shape

def return_objects():
    """Method returns an array of images, one image per object."""


def save_objects(dir_path):
    """Method saves images, one image per object, to files."""
