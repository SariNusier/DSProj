"""
The code below was made with the help of the tutorials by Martin Gorner on MNIST classifier.
For more information please go to
https://github.com/martin-gorner/tensorflow-mnist-tutorial/
"""


import tensorflow as tf
from imgpreproc import reading
from imgpreproc import resizing

ITERATIONS = 100000
DROP_OUT_KEEP = 0.75
l1 = 200
l2 = 100


def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def init_biases(shape, name):
    return tf.Variable(tf.zeros([shape]))


def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden, b1, b2, b3):
    with tf.name_scope("layer1"):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.sigmoid(tf.matmul(X, w_h) + b1)

    with tf.name_scope("layer2"):
        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.sigmoid(tf.matmul(h, w_h2) + b2)

    with tf.name_scope("layer3"):
        h2 = tf.nn.dropout(h2, p_keep_hidden)
        return tf.matmul(h2, w_o) + b3


train_X, test_X, train_Y, test_Y = reading.get_data_tt("/home/sari/data/sorted/", test_size=0.2, resize_method=resizing.RESIZE_BICUBIC,
                                                       labels_format=reading.LABELS_DNN)


print "TRAIN DATA SHAPE: "+ str(train_X.shape)
print "TEST DATA SHAPE: "+ str(test_X.shape)
X = tf.placeholder("float", [None, 784], name="X")
Y = tf.placeholder("float", [None, 4], name="Y")

w_h = init_weights((784, l1), "w_h")
w_h2 = init_weights((l1, l2), "w_h2")
w_o = init_weights((l2, 4), "w_o")
b1 = init_biases(l1, "b1")
b2 = init_biases(l2, "b2")
b3 = init_biases(4, "b3")

tf.summary.histogram("w_h_summ", w_h)
tf.summary.histogram("w_h2_summ", w_h2)
tf.summary.histogram("w_o_summ", w_o)

p_keep_input = tf.placeholder("float", name="p_keep_input")
p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")

py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden, b1, b2, b3)

with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.GradientDescentOptimizer(0.0002).minimize(cost)

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
    for i in range(ITERATIONS):

        batch_X, batch_Y = reading.get_sample(200, train_X, train_Y)
        cs, _ = sess.run([cost, train_op], feed_dict={X: batch_X, Y: batch_Y,
                                      p_keep_input: 1, p_keep_hidden: DROP_OUT_KEEP})
        summary, acc = sess.run([merged, acc_op], feed_dict={X: test_X, Y: test_Y,
                                                             p_keep_input: 1.0, p_keep_hidden: 1.0})
        writer.add_summary(summary, i)
        print cs
        crs_ent.append(cs)
        print i, acc
        accuracies.append(acc)

    """
    class_step = py_x.eval(feed_dict={X: test_X, p_keep_input: 1, p_keep_hidden: 1})
    results = []
    for r in class_step:
        results.append(r.argmax())

    print metrics.accuracy_score(results, reading.convert_labels(test_Y))
    print metrics.accuracy_score(reading.convert_labels(test_Y), results)
    print metrics.classification_report(reading.convert_labels(test_Y), results)
    report = metrics.classification_report(reading.convert_labels(test_Y), results)
    conf_mat = metrics.confusion_matrix(reading.convert_labels(test_Y), results)
    to_return = {'accuracy': accuracies, 'cross_entropy': crs_ent, 'report': report, 'conf_mat': conf_mat}

    pickle.dump(to_return, open("/home/sari/workspace/DSProj/code/main/results/test2.p", "wb"))
    """
