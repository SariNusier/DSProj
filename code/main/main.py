import imgpreproc.resizing as resizing
import imgpreproc.reading as reading
import numpy as np
from PIL import Image
from mltools import svmclassifier
from mltools import testinception
import cPickle as pickle
import matplotlib.pyplot as plt


def svm_classification():
    svm_cls = svmclassifier.SVMClassifier()


def predict_svm():
    print "HELLO"
    svm_clf_poly = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True)
    svm_clf_rbf = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, kernel='rbf', scaling=True)
    svm_clf_linear = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, kernel='linear')
    svm_clf_sigmoid = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, kernel='sigmoid')
    svm_clf_poly_padl = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True,
                                                    resize_method=resizing.RESIZE_BILINEAR)
    svm_clf_poly_padc = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True,
                                                    resize_method=resizing.RESIZE_BICUBIC)
    svm_clf_poly_pad = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, resize_method=resizing.RESIZE_PW)
    print svm_clf_rbf.accuracy
    print svm_clf_sigmoid.accuracy
    print svm_clf_linear.accuracy
    print svm_clf_poly.accuracy
    print svm_clf_poly_padl.accuracy
    print svm_clf_poly_padc.accuracy
    print svm_clf_poly_pad.accuracy
    """
    svm_clf_rbf = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, kernel='rbf', scaling=True)
    svm_clf_linear = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, kernel='linear')
    svm_clf_sigmoid = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, kernel='sigmoid')
    """
    """
    data, images = reading.get_test_data()
    data = data.reshape((data.shape[0], 30, 30))
    print "SHAPE OF DATA: " + str(data.shape)
    print "SHAPE OF DATA: " + str(data[0].reshape(1, -1).shape)
    for i, d in enumerate(data):
        # print "Saving: " + str(i)
        cls = svm_clf.predict(d.reshape(1, -1))
        if cls == 0:
            Image.fromarray(images[i]).save("/home/sari/data/svm/whole/%d.tiff" % i)
        if cls == 1:
            Image.fromarray(images[i]).save("/home/sari/data/svm/noise/%d.tiff" % i)
        if cls == 2:
            Image.fromarray(images[i]).save("/home/sari/data/svm/clumps/%d.tiff" % i)
        if cls == 3:
            Image.fromarray(images[i]).save("/home/sari/data/svm/spread/%d.tiff" % i)
    """


def main():
    print "Calling"
    predict_svm()


if __name__ == '__main__':
    main()
