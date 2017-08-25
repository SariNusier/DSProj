import tensorflow as tf
import os
from sklearn import metrics


def read_local(path):
    """ Gets images from local directory and returns them as a list of numpy arrays"""
    files = os.listdir(path)
    image_data = []
    for f in files:
        image_data.append(tf.gfile.FastGFile(os.path.join(path, f), 'rb').read())

    return image_data


def run(path="/home/sari/data/test-inception/test/"):
    imgs_whole = read_local(os.path.join(path, "whole/"))
    imgs_noise = read_local(os.path.join(path, "noise"))
    imgs_clumps = read_local(os.path.join(path, "clumps"))
    imgs_spread = read_local(os.path.join(path, "spread"))
    true_labels = []
    predicted_labels = []
    image_data = imgs_whole + imgs_noise + imgs_clumps + imgs_spread

    for i in imgs_whole:
        true_labels.append("whole")
    for i in imgs_noise:
        true_labels.append("noise")
    for i in imgs_clumps:
        true_labels.append("clumps")
    for i in imgs_spread:
        true_labels.append("spread")

    label_lines = [line.rstrip() for line in tf.gfile.GFile("/home/sari/Desktop/tftest/retrained_labels.txt")]

    with tf.gfile.FastGFile("/home/sari/Desktop/tftest/retrained_graph.gp", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for img in image_data:
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': img})

            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            predicted_labels.append(label_lines[top_k[0]])
        print predicted_labels
        print metrics.accuracy_score(predicted_labels, true_labels)
        print metrics.classification_report(predicted_labels, true_labels)
        print metrics.confusion_matrix(predicted_labels, true_labels)

