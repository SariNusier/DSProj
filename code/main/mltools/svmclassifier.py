import numpy as np
from sklearn import svm
from imgpreproc import reading
from imgpreproc import resizing

TEST_SIZE = 0.20
KERNEL_RBF = 'rbf'
KERNEL_POLY = 'poly'
KERNEL_SIGMOID = 'sigmoid'
KERNEL_LINEAR = 'linear'

KERNEL_DEFAULT = KERNEL_POLY


class SVMClassifier:
    def __init__(self, test_size=TEST_SIZE, kernel=KERNEL_DEFAULT, gamma='auto', C=100, probability=False,
                 test_mode=False, scaling=True, resize_method=resizing.RESIZE_NEAREST):
        self.test_size = test_size
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.probability = probability
        self.scaling = scaling
        self.resize_method = resize_method
        if test_mode:
            self.train_X, self.test_X, self.train_Y, self.test_Y = reading.get_data_tt(test_size, resize_method=self.resize_method)
            self.data = self.train_X
            self.labels = self.train_Y

        else:
            self.data, self.labels = reading.get_data(labels_format=reading.LABELS_NORMAL, resize_method=self.resize_method)

        self.resh = self.scale_input(np.reshape(self.data, [-1, 900]))
        if test_mode:
            self.test_X = self.scale_input(np.reshape(self.test_X, [-1, 900]))

        self.clf = svm.SVC(gamma=self.gamma, C=self.C, probability=self.probability, kernel=self.kernel, degree=2)
        self.clf.fit(self.resh, self.labels)

        if test_mode:
            count = 0.0
            self.results = []
            for idx, inst in enumerate(self.test_X):
                prediction = self.predict(inst.reshape(1, -1))[0]
                if prediction == self.test_Y[idx]:
                    count = count + 1

                self.results.append(count / (idx + 1))

    def predict(self, img):
        img = img.reshape(-1)
        return self.clf.predict(img)

    def scale_input(self, data):
        if self.scaling:
            return np.array([d / float(255) for d in data])
        else:
            return data
