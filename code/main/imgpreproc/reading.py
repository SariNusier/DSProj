from __future__ import print_function

from PIL import Image
import numpy as np
from HTMLParser import HTMLParser
import urllib2
import numpy as np
import sys
import cStringIO


class MyHTMLParser(HTMLParser):
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


def read_images():
    file_names = ["/home/sari/Desktop/test-test-3/Object_Test_%d.tif" % i for i in range(1, 207)]
    images = []
    for fn in file_names:
        images.append(Image.open(fn))
    return images


def read_from_server():
    url_base = "http://10.200.102.18/"
    url_dir = "G179-dataset/"
    all_images = urllib2.urlopen(url_base + url_dir).read()

    parser = MyHTMLParser()
    parser.feed(all_images)
    data = parser.data
    imgs = []

    print("Started Download!")
    i = 1

    for d in data:
        print("\rProgress: %d/%d " % (i, len(data)), end='')
        dl_img = urllib2.urlopen(url_base + url_dir + d).read()
        asd = cStringIO.StringIO(dl_img)
        img = Image.open(asd)
        imgs.append(np.array(img))
        i = i+1
        if i == 10:
            break

    return imgs


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
