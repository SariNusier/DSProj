import reading
import resizing
from PIL import Image

def convert():
    """Converts tiffs into jpeg, preserving the label given by the directory"""
    imgs, labels = reading.get_data(resize_method=resizing.RESIZE_NONE)
    train_X, test_X, train_Y, test_Y = reading.get_data_tt("/home/sari/data/sorted/",
                                                           test_size=0.2,
                                                           resize_method=resizing.RESIZE_NONE)


    for i, img in enumerate(train_X):
        if train_Y[i] == 0:
            Image.fromarray(img).save("/home/sari/data/test-inception/train/whole/%d.jpg" % i)
        elif train_Y[i] == 1:
            Image.fromarray(img).save("/home/sari/data/test-inception/train/noise/%d.jpg" % i)
        elif train_Y[i] == 2:
            Image.fromarray(img).save("/home/sari/data/test-inception/train/clumps/%d.jpg" % i)
        elif train_Y[i] == 3:
            Image.fromarray(img).save("/home/sari/data/test-inception/train/spread/%d.jpg" % i)

    for i, img in enumerate(test_X):
        if test_Y[i] == 0:
            Image.fromarray(img).save("/home/sari/data/test-inception/test/whole/%d.jpg" % i)
        elif test_Y[i] == 1:
            Image.fromarray(img).save("/home/sari/data/test-inception/test/noise/%d.jpg" % i)
        elif test_Y[i] == 2:
            Image.fromarray(img).save("/home/sari/data/test-inception/test/clumps/%d.jpg" % i)
        elif test_Y[i] == 3:
            Image.fromarray(img).save("/home/sari/data/test-inception/test/spread/%d.jpg" % i)
