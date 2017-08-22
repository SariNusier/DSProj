import tensorflow as tf
from imgpreproc import reading
from imgpreproc import resizing
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

l1 = 200
l2 = 100
l3 = 60
l4 = 30


def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def init_biases(shape, name):
    return tf.Variable(tf.zeros([shape]))


def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden, b1, b2, b3):
    with tf.name_scope("layer1"):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.matmul(X, w_h) + b1)

    with tf.name_scope("layer2"):
        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2) + b2)

    with tf.name_scope("layer3"):
        h2 = tf.nn.dropout(h2, p_keep_hidden)
        return tf.matmul(h2, w_o) + b3


train_X, test_X, train_Y, test_Y = reading.get_data_tt(test_size=0.2, resize_method=resizing.RESIZE_NEAREST,
                                                       labels_format=reading.LABELS_TF)


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
    # train_op = tf.train.RMSPropOptimizer(0.002, 0.9).minimize(cost)
    train_op = tf.train.AdamOptimizer(0.0002).minimize(cost)

    tf.summary.scalar("cost", cost)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float"))
    tf.summary.scalar("accuracy", acc_op)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)
    merged = tf.summary.merge_all()
    tf.initialize_all_variables().run()

    for i in range(100000):

        batch_X, batch_Y = reading.get_sample(100, train_X, train_Y)
        sess.run(train_op, feed_dict={X: batch_X, Y: batch_Y,
                                      p_keep_input: 1, p_keep_hidden: 0.75})
        summary, acc = sess.run([merged, acc_op], feed_dict={X: test_X, Y: test_Y,
                                                             p_keep_input: 1.0, p_keep_hidden: 1.0})
        writer.add_summary(summary, i)
        print i, acc

print train_X.shape
print test_X.shape
print train_Y
print test_Y
