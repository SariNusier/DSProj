import numpy as np
from sklearn import svm
from imgpreproc import reading

TEST_SIZE = 0.20
KERNEL_DEFAULT = 'poly'


class SVMClassifier:
    def __init__(self, test_size=TEST_SIZE, kernel=KERNEL_DEFAULT, gamma='auto', C=100, probability=True, test_mode=False):
        self.test_size = test_size
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.probability = probability
        if test_mode:
            self.train_X, self.test_X, self.train_Y, self.test_Y = reading.get_data_tt(test_size)
            self.data = self.train_X
            self.labels = self.train_Y

        else:
            self.data, self.labels = reading.get_data(labels_format=reading.LABELS_NORMAL)

        self.resh = np.reshape(self.data, [-1, 900])
        self.test_X = np.reshape(self.test_X, [-1, 900])

        self.clf = svm.SVC(gamma=self.gamma, C=self.C, probability=self.probability, kernel=self.kernel, degree=2)
        self.clf.fit(self.resh, self.labels)

        if test_mode:
            count = 0.0
            self.results = []
            for idx, inst in enumerate(self.test_X):
                prediction = self.predict(inst.reshape(1, -1))[0]
                if prediction == self.test_Y[idx]:
                    count = count + 1

                self.results.append(count / (idx+1))

    def predict(self, img):
        img = img.reshape(-1)
        return self.clf.predict(img)
