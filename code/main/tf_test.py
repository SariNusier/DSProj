import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def main():
    K = 200
    L = 100
    M = 60
    N = 30

    W1 = tf.Variable(tf.truncated_normal([28*28, K], stddev=0.1))
    B1 = tf.Variable(tf.zeros([K]))

    W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
    B2 = tf.Variable(tf.zeros([L]))

    W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
    B3 = tf.Variable(tf.zeros([M]))

    W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
    B4 = tf.Variable(tf.zeros([N]))

    W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    B5 = tf.Variable(tf.zeros([10]))

    X = tf.placeholder(tf.float32, [None, 784])

    init = tf.initialize_all_variables()

    Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)

    Y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

    is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(0.003)
    train_step = optimizer.minimize(cross_entropy)

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_X, batch_Y = mnist.train.next_batch(100)
        train_data = {X: batch_X, Y_: batch_Y}

        sess.run(train_step, feed_dict=train_data)

        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print "Train Accuracy: "+str(a)
        # print "Train Cross entropy: "+str(c)

        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print "Test Accuracy: "+str(a)
        # print "Test Cross entropy: "+str(c)

if __name__ == "__main__":
    main()
