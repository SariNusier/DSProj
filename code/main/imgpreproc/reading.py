from __future__ import print_function

from PIL import Image
import numpy as np
from HTMLParser import HTMLParser
import urllib2
import numpy as np
from imgpreproc import resizing
import sys
import cStringIO
import os
from sklearn.cross_validation import train_test_split

LABELS_TF = 0
LABELS_NORMAL = 1



class ImagesHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.prev_tag = ""
        self.data = []

    def handle_starttag(self, tag, attrs):
        # print "Encountered a start tag:", tag
        self.prev_tag = tag

    def handle_data(self, data):
        # print "Encountered some data  :", data
        if self.prev_tag == 'a' and data.strip() != "":
            self.data.append(urllib2.quote(data))


def read_from_server(url_base="http://10.200.102.18/", url_dir="G179-dataset/"):
    """ Gets images using http and returns them as a list of numpy arrays"""

    all_images = urllib2.urlopen(url_base + url_dir).read()

    parser = ImagesHTMLParser()
    parser.feed(all_images)
    data = parser.data
    imgs = []

    print("Found %d images!" % len(data))
    print("Started Download!")
    i = 1

    for d in data:
        print("\rProgress: %d/%d " % (i, len(data)), end='')
        dl_img = urllib2.urlopen(url_base + url_dir + d).read()
        asd = cStringIO.StringIO(dl_img)
        img = Image.open(asd)
        imgs.append(np.array(img))
        i = i + 1

    return imgs


def read_local(path):
    """ Gets images from local directory and returns them as a list of numpy arrays"""
    files = os.listdir(path)
    imgs = []
    for f in files:
        if f.endswith(".tiff") or f.endswith(".tif"):
            img = Image.open(os.path.join(path, f))
            imgs.append(np.array(img))
            # imgs.append(img)
    return imgs


def get_data(labels_format=LABELS_NORMAL, resize_method=resizing.RESIZE_NONE):
    imgs_whole = read_local("/home/sari/data/sorted/whole")
    imgs_noise = read_local("/home/sari/data/sorted/noise")
    imgs_clumps = read_local("/home/sari/data/sorted/clumps")
    imgs_spread = read_local("/home/sari/data/sorted/spread")

    # Move resizing method from here
    """
    res_whole = np.array([resizing.resize(img, 30, 30, mode=resize_method) for img in imgs_whole])
    res_noise = np.array([resizing.resize(img, 30, 30, mode=resize_method) for img in imgs_noise])
    res_clumps = np.array([resizing.resize(img, 30, 30, mode=resize_method) for img in imgs_clumps])
    res_spread = np.array([resizing.resize(img, 30, 30, mode=resize_method) for img in imgs_spread])
    """
    res_whole = resizing.resize(imgs_whole, 28, 28, mode=resize_method)
    res_noise = resizing.resize(imgs_noise, 28, 28, mode=resize_method)
    res_clumps = resizing.resize(imgs_clumps, 28, 28, mode=resize_method)
    res_spread = resizing.resize(imgs_spread, 28, 28, mode=resize_method)
    labels = []
    if labels_format == LABELS_TF:
        for i in res_whole:
            labels.append([1, 0, 0, 0])
        for i in res_noise:
            labels.append([0, 1, 0, 0])
        for i in res_clumps:
            labels.append([0, 0, 1, 0])
        for i in res_spread:
            labels.append([0, 0, 0, 1])
    elif labels_format == LABELS_NORMAL:
        for i in res_whole:
            labels.append(0)
        for i in res_noise:
            labels.append(1)
        for i in res_clumps:
            labels.append(2)
        for i in res_spread:
            labels.append(3)

    res_images = np.append(res_whole, res_noise, axis=0)
    res_images = np.append(res_images, res_clumps, axis=0)
    res_images = np.append(res_images, res_spread, axis=0)
    print("Shape in reading"+str(res_images.shape))
    if labels_format == LABELS_TF:
        return res_images.reshape((res_images.shape[0], 28, 28, 1)), np.array(labels)
        # return np.reshape(res_images, [-1, 784]), np.array(labels)
    if labels_format == LABELS_NORMAL:
        return res_images, labels


def get_data_tt(test_size, labels_format=LABELS_NORMAL, resize_method=resizing.RESIZE_NONE):
    data, labels = get_data(labels_format, resize_method)
    print(data.shape)
    return train_test_split(data, labels, test_size=test_size, random_state=42)


def get_test_data():
    images = read_local("/home/sari/data/CP_cropped_1")
    res_images = np.array([np.array(Image.fromarray(img).resize((30, 30), Image.BICUBIC)) for img in images])
    print(res_images.shape)

    return res_images.reshape((res_images.shape[0], 30, 30, 1)), images


def get_data_sample(count):
    data, labels = get_data()
    idx = np.random.choice(np.arange(len(data)), count, replace=False)
    return data[idx], labels[idx]

def get_sample(count, data, labels):
    idx = np.random.choice(np.arange(len(data)), count, replace=False)
    return data[idx], labels[idx]


def get_all_data():
    images = read_local("/home/sari/data/CP_cropped_0")
    return images
