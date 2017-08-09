import imgpreproc.resizing as resizing
import imgpreproc.reading as reading
import numpy as np
from PIL import Image


def main():
    images = reading.read_from_server()
    for i in images:
        Image.fromarray(i).show()

if __name__ == '__main__':
    main()
