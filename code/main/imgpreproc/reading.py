from PIL import Image
import numpy as np


def read_images():
    file_names = ["/home/sari/Desktop/test-test-3/Object_Test_%d.tif" % i for i in range(1, 207)]
    images = []
    for fn in file_names:
        images.append(Image.open(fn))
    return images

def read_from_server():
    :q
    

def get_data():
    images = read_images()
    res_images = np.array([np.array(img.resize((28, 28))) for img in images])
    classes = [2, 0, 1, 2, 2,
               1, 1, 2, 2, 0,
               1, 1, 2, 2, 2,
               2, 0, 2, 2, 2,
               2, 2, 2, 2, 0,
               1, 1, 2, 2, 2,
               0, 0, 1, 1, 2,
               2, 2, 2, 2, 2,
               2, 2, 2, 0, 0,
               1, 2, 0, 0, 2,
               2, 0, 2, 0, 1,
               1, 0, 0, 1, 0,
               2, 2, 0, 0, 0,
               2, 0, 0, 0, 0,
               2, 2, 2, 2, 2,
               2, 2, 0, 0, 2,
               2, 2, 0, 2, 2,
               2, 2, 2, 1, 1,
               2, 2, 2, 0, 0,
               2, 2, 1, 0, 0,
               1, 0, 0, 0, 0]
    parsed = []
    for c in classes:
        if c == 0:
            parsed.append([1, 0, 0])
        elif c == 1:
            parsed.append([0, 1, 0])
        elif c == 2:
            parsed.append([0, 0, 1])

    return res_images.reshape((res_images.shape[0], 28, 28, 1))[:105], np.array(parsed)


def get_test_data():
    images = read_images()
    res_images = np.array([np.array(img.resize((28, 28))) for img in images])

    return res_images.reshape((res_images.shape[0], 28, 28, 1))[106:206]
