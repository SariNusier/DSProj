import imgpreproc.resizing as resizing
import imgpreproc.reading as reading
import numpy as np
from PIL import Image
from mltools import svmclassifier
import matplotlib.pyplot as plt


def svm_classification():
    svm_cls = svmclassifier.SVMClassifier()
    svm_cls


def predict_svm():
    svm_clf_poly = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True)
    svm_clf_rbf = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, kernel='rbf', scaling=True)
    svm_clf_linear = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, kernel='linear')
    svm_clf_sigmoid = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, kernel='sigmoid')
    svm_clf_poly_padl = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, resize_method=resizing.RESIZE_BILINEAR)
    svm_clf_poly_padc = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, resize_method=resizing.RESIZE_BICUBIC)
    svm_clf_poly_pad = svmclassifier.SVMClassifier(test_size=0.20, test_mode=True, resize_method=resizing.RESIZE_PW)
    print svm_clf_rbf.results[-1]
    print svm_clf_sigmoid.results[-1]
    print svm_clf_linear.results[-1]
    print svm_clf_poly.results[-1]
    print svm_clf_poly_padl.results[-1]
    print svm_clf_poly_padc.results[-1]
    print svm_clf_poly_pad.results[-1]
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

def resizing_t(mode):
    img, _ = reading.get_data(resize_method=mode)
    img = img[0]
    print img
    print img.shape

def main():
    """
    images = reading.read_from_server(url_dir="Cellprofiler%20workshop/Bob/FN1/")
    for i in images:
        Image.fromarray(i).show()
    # images = reading.read_local("/home/sari/Desktop/sk_cp_output")
    images = reading.read_local("/home/sari/Desktop/data")
    img = images[0]
    resized_img = resizing.resize2(img, 3000, 3000)
    print type(resized_img)
    # arr = np.array([[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, 24, 25, 26, 27], [31, 32, 33, 34, 35, 36, 37]])
    # resized_arr = resizing.sliding_window(arr, 3, 3)

    print "This is the number of results" + str(len(resized_img))
    if type(resized_img) == np.ndarray:
        Image.fromarray(resized_img).show()
    else:
        for i in resized_img:
            Image.fromarray(i).show()

    """
    """
    resized_img = resizing.resize(img, 100, 100, Image.NEAREST)
    resized_img1 = resizing.resize(img, 100, 100, Image.BILINEAR)
    resized_img2 = resizing.resize(img, 100, 100, Image.BICUBIC)
    resized_img3 = resizing.resize(img, 100, 100, Image.LANCZOS)
    Image.fromarray(img).save("/home/sari/Desktop/orig.tiff")
    Image.fromarray(resized_img).show()
    Image.fromarray(resized_img).save("/home/sari/Desktop/nn.tiff")
    Image.fromarray(resized_img1).show()
    Image.fromarray(resized_img1).save("/home/sari/Desktop/bl.tiff")
    Image.fromarray(resized_img2).show()
    Image.fromarray(resized_img2).save("/home/sari/Desktop/bic.tiff")
    Image.fromarray(resized_img3).show()
    Image.fromarray(resized_img3).save("/home/sari/Desktop/lanc.tiff")
    """
    # something()
    """
    images_raw, labels = reading.get_data()
    for idx, img in enumerate(images_raw):
        cls = labels[idx]
        if cls == 0:
            Image.fromarray(img).save("/home/sari/data/testtf/whole/%d.jpg" % idx)
        if cls == 1:
            Image.fromarray(img).save("/home/sari/data/testtf/noise/%d.jpg" % idx)
        if cls == 2:
            Image.fromarray(img).save("/home/sari/data/testtf/clumps/%d.jpg" % idx)
        if cls == 3:
            Image.fromarray(img).save("/home/sari/data/testtf/spread/%d.jpg" % idx)
    # predict_svm()
    images = reading.read_local("/home/sari/data/CP_cropped_0")
    for i, img in enumerate(images):
        Image.fromarray(img).save("/home/sari/data/inception0/%d.jpg" % i)
    print len(images)
    """
    from mltools import testinception

    testinception.run()

if __name__ == '__main__':
    main()
