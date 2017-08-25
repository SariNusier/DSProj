from __future__ import print_function

from PIL import Image
from HTMLParser import HTMLParser
import urllib2
import numpy as np
from imgpreproc import resizing
import cStringIO
import os
from sklearn.cross_validation import train_test_split

LABELS_TF = 0
LABELS_NORMAL = 1
LABELS_DNN = 2


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
    return imgs


def get_data(path, labels_format=LABELS_NORMAL, resize_method=resizing.RESIZE_NONE):
    imgs_whole = read_local(os.path.join(path, "whole/"))
    imgs_noise = read_local(os.path.join(path, "noise/"))
    imgs_clumps = read_local(os.path.join(path, "clumps/"))
    imgs_spread = read_local(os.path.join(path, "spread/"))

    res_whole = resizing.resize(imgs_whole, 28, 28, mode=resize_method)
    res_noise = resizing.resize(imgs_noise, 28, 28, mode=resize_method)
    res_clumps = resizing.resize(imgs_clumps, 28, 28, mode=resize_method)
    res_spread = resizing.resize(imgs_spread, 28, 28, mode=resize_method)
    labels = []
    if labels_format == LABELS_TF or labels_format == LABELS_DNN:
        for _ in res_whole:
            labels.append([1, 0, 0, 0])
        for _ in res_noise:
            labels.append([0, 1, 0, 0])
        for _ in res_clumps:
            labels.append([0, 0, 1, 0])
        for _ in res_spread:
            labels.append([0, 0, 0, 1])
    elif labels_format == LABELS_NORMAL:
        for _ in res_whole:
            labels.append(0)
        for _ in res_noise:
            labels.append(1)
        for _ in res_clumps:
            labels.append(2)
        for _ in res_spread:
            labels.append(3)

    res_images = np.append(res_whole, res_noise, axis=0)
    res_images = np.append(res_images, res_clumps, axis=0)
    res_images = np.append(res_images, res_spread, axis=0)
    if labels_format == LABELS_TF:
        return res_images.reshape((res_images.shape[0], 28, 28, 1)), np.array(labels)
        # return np.reshape(res_images, [-1, 784]), np.array(labels)
    if labels_format == LABELS_DNN:
        return np.reshape(res_images, [-1, 784]), np.array(labels)
    if labels_format == LABELS_NORMAL:
        return res_images, labels


def get_data_tt(path, test_size, labels_format=LABELS_NORMAL, resize_method=resizing.RESIZE_NONE):
    data, labels = get_data(path, labels_format, resize_method)
    return train_test_split(data, labels, test_size=test_size, random_state=42)


def get_sample(count, data, labels):
    idx = np.random.choice(np.arange(len(data)), count, replace=False)
    return data[idx], labels[idx]


def convert_labels(labels):
    to_ret = []
    for i in labels:
        to_ret.append(i.argmax())

    return to_ret

