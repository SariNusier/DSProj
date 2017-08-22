import tensorflow as tf
import sys

def run():

    image_data = []
    for i in range(3506):
        image_data.append(tf.gfile.FastGFile("/home/sari/data/inception0/%d.jpg"%i, 'rb').read())

    label_lines = [line.rstrip() for line in tf.gfile.GFile("/home/sari/workspace/tftest/retrained_labels.txt")]

    with tf.gfile.FastGFile("/home/sari/workspace/tftest/retrained_graph.gp", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for img in image_data:
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': img})


            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            print label_lines[top_k[0]]
            print predictions[0][top_k[0]]
        """
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            print human_string
        """