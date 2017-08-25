import numpy as np
from sklearn import svm
from imgpreproc import reading
from imgpreproc import resizing
from sklearn import metrics

TEST_SIZE = 0.20
KERNEL_RBF = 'rbf'
KERNEL_POLY = 'poly'
KERNEL_SIGMOID = 'sigmoid'
KERNEL_LINEAR = 'linear'
KERNEL_LIST = [KERNEL_RBF, KERNEL_POLY, KERNEL_SIGMOID, KERNEL_LINEAR]
KERNEL_DEFAULT = KERNEL_POLY


class SVMClassifier:
    def __init__(self, test_size=TEST_SIZE, kernel=KERNEL_DEFAULT, gamma='auto', C=500, probability=False,
                 test_mode=False, scaling=True, resize_method=resizing.RESIZE_NEAREST):
        self.test_size = test_size
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.probability = probability
        self.scaling = scaling
        self.resize_method = resize_method

        # Metrics
        self.accuracy = 0
        self.average_precision = 0
        self.summary = 0
        self.conf_matrix = 0
        self.recall = 0
        self.f1_score = 0

        if test_mode:
            self.train_X, self.test_X, self.train_Y, self.test_Y = reading.get_data_tt("/home/sari/data/sorted/",
                                                                                       test_size,
                                                                                       resize_method=self.resize_method)
            self.data = self.train_X
            self.labels = self.train_Y

        else:
            self.data, self.labels = reading.get_data("/home/sari/data/sorted/", labels_format=reading.LABELS_NORMAL,
                                                      resize_method=self.resize_method)

        self.resh = self.scale_input(np.reshape(self.data, [-1, 784]))
        if test_mode:
            self.test_X = self.scale_input(np.reshape(self.test_X, [-1, 784]))

        self.clf = svm.SVC(gamma=self.gamma, C=self.C, probability=self.probability, kernel=self.kernel, degree=2)
        self.clf.fit(self.resh, self.labels)

        if test_mode:
            predictions = []
            for idx, inst in enumerate(self.test_X):
                a = np.reshape(inst, (1, -1))
                prediction = self.predict(a)[0]
                predictions.append(prediction)

            self.accuracy = metrics.accuracy_score(self.test_Y, predictions)
            self.recall = metrics.recall_score(self.test_Y, predictions, average='macro')
            self.f1_score = metrics.f1_score(self.test_Y, predictions, average='weighted')
            self.summary = metrics.classification_report(self.test_Y, predictions)
            self.cohen_k = metrics.cohen_kappa_score(self.test_Y, predictions)
            self.conf_matrix = metrics.confusion_matrix(self.test_Y, predictions)

    def predict(self, img):
        return self.clf.predict(img)

    def scale_input(self, data):
        if self.scaling:
            return np.array([d / float(255) for d in data])
        else:
            return data
