import tensorflow as tf
from PIL import Image
from imgpreproc import reading
from imgpreproc import resizing
import matplotlib.pyplot as plt
import math
from sklearn import metrics
import cPickle as pickle

l1 = 4
l2 = 8
l3 = 12
l4 = 200


def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def init_biases(shape, name):
    return tf.Variable(tf.ones([shape]) / 10)


def model(X, w_c1, w_c2, w_c3, w_f4, w_f5, p_keep_input, p_keep_hidden, b1, b2, b3, b4, b5):
    with tf.name_scope("layer1"):
        Y1 = tf.nn.relu(tf.nn.conv2d(X, w_c1, strides=[1, 1, 1, 1], padding='SAME') + b1)

    with tf.name_scope("layer2"):
        Y2 = tf.nn.relu(tf.nn.conv2d(Y1, w_c2, strides=[1, 2, 2, 1], padding='SAME') + b2)

    with tf.name_scope("layer3"):
        Y3 = tf.nn.relu(tf.nn.conv2d(Y2, w_c3, strides=[1, 2, 2, 1], padding='SAME') + b3)

    with tf.name_scope("layer4"):
        print Y3.shape
        YY = tf.reshape(Y3, shape=[-1, 7 * 7 * l3])

        Y4 = tf.nn.relu(tf.matmul(YY, w_f4) + b4)
        YY4 = tf.nn.dropout(Y4, p_keep_hidden)

    with tf.name_scope("layer5"):
        return tf.matmul(YY4, w_f5) + b5


train_X, test_X, train_Y, test_Y = reading.get_data_tt(test_size=0.2, resize_method=resizing.RESIZE_BILINEAR,
                                                       labels_format=reading.LABELS_TF)

print train_X.shape

print "TRAIN DATA SHAPE: " + str(train_X.shape)
print "TEST DATA SHAPE: " + str(test_X.shape)
X = tf.placeholder("float", [None, 28, 28, 1], name="X")
Y = tf.placeholder("float", [None, 4], name="Y")

w_c1 = init_weights((5, 5, 1, l1), "w_c1")
w_c2 = init_weights((5, 5, l1, l2), "w_c2")
w_c3 = init_weights((4, 4, l2, l3), "w_c3")
w_f4 = init_weights((7 * 7 * l3, l4), "w_f4")
w_f5 = init_weights((l4, 4), "w_f5")
b1 = init_biases(l1, "b1")
b2 = init_biases(l2, "b2")
b3 = init_biases(l3, "b3")
b4 = init_biases(l4, "b4")
b5 = init_biases(4, "b5")

tf.summary.histogram("w_h_summ", w_c1)
tf.summary.histogram("w_h2_summ", w_c2)
tf.summary.histogram("w_o_summ", w_c3)

p_keep_input = tf.placeholder("float", name="p_keep_input")
p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")
l_rate = tf.placeholder("float", name="l_rate")

py_x = model(X, w_c1, w_c2, w_c3, w_f4, w_f5, p_keep_input, p_keep_hidden, b1, b2, b3, b4, b5)
accuracy = []
costs = []
accuracy_train = []
costs_train = []
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    # train_op = tf.train.RMSPropOptimizer(0.002, 0.9).minimize(cost)
    train_op = tf.train.AdamOptimizer(l_rate).minimize(cost)

    tf.summary.scalar("cost", cost)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float"))
    tf.summary.scalar("accuracy", acc_op)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)
    merged = tf.summary.merge_all()
    tf.initialize_all_variables().run()
    crs_ent = []
    accuracies = []
    for i in range(50000):
        max_learning_rate = 0.003
        min_learning_rate = 0.0001
        decay_speed = 2000.0
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)
        batch_X, batch_Y = reading.get_sample(100, train_X, train_Y)
        acc_train, c_train, _= sess.run([acc_op,cost,train_op], feed_dict={X: batch_X, Y: batch_Y,
                                      p_keep_input: 1, p_keep_hidden: 0.75, l_rate: learning_rate})
        summary, acc, c = sess.run([merged, acc_op, cost], feed_dict={X: test_X, Y: test_Y,
                                                             p_keep_input: 1.0, p_keep_hidden: 1})
        writer.add_summary(summary, i)
        accuracy.append(acc)
        costs.append(c)
        costs_train.append(c_train)
        accuracy_train.append(acc_train)

        print i, acc_train, c_train
        print acc, c
        accuracies.append(acc)
        crs_ent.append(c_train)

    class_step = py_x.eval(feed_dict={X: test_X, p_keep_hidden: 1.0, p_keep_input: 1.0})
    results = []
    print class_step
    for r in class_step:
        print r
        results.append(r.argmax())
    print metrics.accuracy_score(results, reading.convert_labels(test_Y))
    print metrics.accuracy_score(reading.convert_labels(test_Y), results)
    print metrics.classification_report(reading.convert_labels(test_Y), results)
    report = metrics.classification_report(reading.convert_labels(test_Y), results)
    conf_mat = metrics.confusion_matrix(reading.convert_labels(test_Y), results)
    to_return = {'accuracy': accuracies, 'cross_entropy': crs_ent, 'report': report, 'conf_mat': conf_mat}

    pickle.dump(to_return, open("/home/sari/workspace/DSProj/code/main/results/CNN6502BC.p", "wb"))


