import tensorflow as tf
from imgpreproc import reading

def initialise(shape, name)
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    with tf.name_scope("layer1"):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.matmul(X, w_h))

    with tf.name_scope("layer2"):
        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2))

    with tf.name_scope("layer3"):
        h2 = tf.nn.dropout(h2, p_keep_hidden)
        return tf.matmul(h2, w_o)


input_data = reading.get_data()

X = tf.placeholder(tf.float32, [None, 30, 30, 1])
Y = tf.placeholder(tf.floate32, [None, 3])