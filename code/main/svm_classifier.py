from PIL import Image
import numpy as np
from sklearn import svm
from sklearn import datasets
from imgpreproc import reading

SAMPLE_SIZE = 500

data, labels = reading.get_data_sample(SAMPLE_SIZE)
digits = datasets.load_digits()
ls = []
for l in labels:
    if l[0] == 1:
        ls.append(0)
    if l[1] == 1:
        ls.append(1)
    if l[2] == 1:
        ls.append(2)

print ls
print labels
print digits.data.shape
print digits.images.shape
print digits.target.shape
print labels.shape
print labels.shape
print labels.shape

data = data.reshape((SAMPLE_SIZE, 30, 30))
resh = np.reshape(data, [-1, 900])

clf = svm.SVC(gamma=0.1, C=100., probability=True, kernel='poly', degree=4)
print clf.fit(resh[:400], ls[:400])

count = 0.0
for i in range(401, 501):
    if ls[i] == clf.predict(resh[i])[0]:
        count = count + 1
    else:
        print "Wrong!!!!!!"
    print count / (i - 400)
