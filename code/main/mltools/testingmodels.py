from __future__ import print_function

import numpy as np
import cPickle as pickle
from imgpreproc import reading
from imgpreproc import resizing
from sklearn import metrics
from mltools import svmclassifier
import matplotlib.pyplot as plt

WANTED_KERNEL = svmclassifier.KERNEL_LINEAR


def test_svm():
    # svm_clf_poly = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True)
    classifiers = []
    total = len(resizing.RESIZE_LIST) * len(svmclassifier.KERNEL_LIST) * 5
    i = 1
    for kernel in svmclassifier.KERNEL_LIST:
        if kernel == WANTED_KERNEL:
            for r_method in resizing.RESIZE_LIST:
                for test_size in range(1, 6):
                    c = svmclassifier.SVMClassifier(test_size=test_size / 10.0, test_mode=True, kernel=kernel,
                                                    scaling=True,
                                                    resize_method=r_method)
                    classifiers.append(c)
                    # print("\rProgress: %d/%d " % (i, total), end='')
                    i = i + 1

    acc = {}
    for r in resizing.RESIZE_LIST:
        acc[r] = []

    for c in classifiers:
        if c.kernel == WANTED_KERNEL:
            """
            print ("Resize: " + str(c.resize_method))
            print ("Training size: " + str(c.test_size))
            print (c.accuracy)
            """
            acc[c.resize_method].append(c.accuracy)

            # print svm_clf_rbf.accuracy
    # print type(svm_clf_rbf.summary)
    # print svm_clf_rbf.conf_matrix
    for a in acc:
        print(a)
        print(acc[a])
        if a == resizing.RESIZE_NEAREST:
            l = "Nearest Neighbour"
        if a == resizing.RESIZE_BILINEAR:
            l = "Bilinear interpolation"
        if a == resizing.RESIZE_BICUBIC:
            l = "Bicubic interpolation"
        if a == resizing.RESIZE_PW:
            l = "Pad and Slide"

        # plt.plot([i / 10.0 for i in range(1, 6)], acc[a], label=l)


def test_NN():

    # plt.title("Linear Kernel - Accuracy based on training set size and Resizing method")
    info = pickle.load(open("results/RGDDOBL2.p", "rb"))

    plt.plot(range(len(info['accuracy'])), info['accuracy'])
    plt.show()
    plt.close()
    print(info['cross_entropy'])
    plt.plot(range(len(info['cross_entropy'])), info['cross_entropy'])
    plt.show()

    """
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('%sAccTsRs.png' % k)
    plt.close()
for k in svmclassifier.KERNEL_LIST:
    WANTED_KERNEL = k
    print(k)
    test_svm()
    """

test_NN()