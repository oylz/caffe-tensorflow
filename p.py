#!/usr/bin/python

import tensorflow as tf
from out import AttrNet


data_node = tf.placeholder(tf.float32, shape=(1, 64, 64, 3))
net = AttrNet({'data': data_node})
with tf.Session() as sess:
    output_graph = sess._graph
    net.load(data_path='./out/d.npy', session=sess)
    f = tf.gfile.FastGFile('./out/AttrNet.pb', "w")
    f.write(output_graph.as_graph_def().SerializeToString())
    f.close()


