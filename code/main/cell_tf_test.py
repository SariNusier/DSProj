# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from imgpreproc import reading
from imgpreproc import resizing
from PIL import Image
import numpy as np
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid)      W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid)      W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (sigmoid)      W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (sigmoid)      W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]


X = tf.placeholder(tf.float32, [None, 30, 30, 1])
Y_ = tf.placeholder(tf.float32, [None, 3])

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 200
M = 100
N = 60
O = 30
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([900, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.zeros([M]))
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.zeros([N]))
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.zeros([O]))
W5 = tf.Variable(tf.truncated_normal([O, 3], stddev=0.1))
B5 = tf.Variable(tf.zeros([3]))

# The model
XX = tf.reshape(X, [-1, 900])
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, learning rate = 0.003
learning_rate = 0.003
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.InteractiveSession()
writer = tf.summary.FileWriter("./logs/nn_logs")
merged = tf.summary.merge_all()

sess.run(init)

# You can call this function in a loop to train the model, 100 images at a time
for i in range(1000):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = reading.get_data(labels_format=reading.LABELS_TF)
    print "batch_X: " + str(batch_X.shape)
    print "batch_Y: " + str(batch_X.shape)
    train_data = {X: batch_X, Y_: batch_Y}

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y})
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    print "Accuracy: " + str(a)
    # tf.scalar_summary("accuracy", a)
    # writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph)
    writer.add_graph(sess.graph)
    tf.summary.scalar('accuracy', a)

    test_data = {X: batch_X[100:], Y_: batch_Y[100:]}
    a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    print "Test Accuracy: "+str(a)
    print i

save_path = saver.save(sess, "./my_model_final.ckpt")

batch_X, images = reading.get_test_data()


print batch_X.shape
class_step = Y.eval(feed_dict={X: batch_X})

for i, r in enumerate(class_step):
    print "Saving: " + str(i)
    if r.argmax() == 0:
        Image.fromarray(images[i]).save("/home/sari/data/auto/whole/%d.tiff"%i)
    if r.argmax() == 1:
        Image.fromarray(images[i]).save("/home/sari/data/auto/noise/%d.tiff"%i)
    if r.argmax() == 2:
        Image.fromarray(images[i]).save("/home/sari/data/auto/clumps/%d.tiff"%i)

print class_step
y_pred = sess.run(class_step)
