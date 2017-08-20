import numpy as np
from sklearn import svm
from imgpreproc import reading

SAMPLE_SIZE = 500
KERNEL_DEFAULT = 'poly'


class SVMClassifier:
    def __init__(self, sample_size=SAMPLE_SIZE, kernel=KERNEL_DEFAULT, gamma=0.1, C=100, probability=True):
        self.sample_size = sample_size
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.probability = probability
        self.data, self.labels = reading.get_data(labels_format=reading.LABELS_NORMAL)

        self.data = self.data.reshape((self.data.shape[0], 30, 30))
        self.resh = np.reshape(self.data, [-1, 900])

        self.clf = svm.SVC(gamma=self.gamma, C=self.C, probability=self.probability, kernel=self.kernel, degree=2)
        print self.clf.fit(self.resh, self.labels)
        print

    def predict(self, img):
        img = img.reshape(-1)
        return self.clf.predict(img)


