import imagepreprocessing.resizing as resizing
import numpy as np
from PIL import Image


def main():
    img = resizing.read_image("/home/sari/Desktop/test-test-3/Object_Test_13.tif")
    file_names = ["/home/sari/Desktop/test-test-3/Object_Test_%d.tif"%i for i in range(1, 207)]
    images = []
    for fn in file_names:
        images.append(Image.open(fn))


if __name__ == '__main__':
    main()
