#!/usr/bin/python

import tensorflow as tf
from out import AttrNet
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile

"""data_node = tf.placeholder(tf.float32, shape=(1, 64, 64, 3))
net = AttrNet({'data': data_node})
with tf.Session() as sess:
    net.load(data_path='./out/attrnet.npy', session=sess)
    output_graph = sess._graph
    for node in output_graph.as_graph_def().node:
        print("n================================================n")
        print(node.name)
	#node.device = ""
        print(node) 
        print("u================================================u")
    

    f = tf.gfile.FastGFile('./out/AttrNet.pb', "w")
    f.write(output_graph.as_graph_def().SerializeToString())
    f.close()
"""


def set_attr_dtype(node, key, value):
    try:
        node.attr[key].CopyFrom(value)
    except KeyError:
        pass


def set_attr_tensor(node, key, value, dtype, shape=None):
    try:
        node.attr[key].CopyFrom(tf.AttrValue(
            tensor=tensor_util.make_tensor_proto(value,
                                             dtype=dtype,
                                             shape=shape)))
    except KeyError:
        pass

def DD(session, input_graph_def, clear_devices):
    if clear_devices:
        print("begin nodename============")
        for node in input_graph_def.node:
            print(node.name)
            print(node)
            #xyz20171022 node.device = ""
        print("end nodename============")
    _ = tf.import_graph_def(input_graph_def, name="")

    found_variables = {}
    for node in input_graph_def.node:
        if node.op == "Assign":
            variable_name = node.input[0]
            found_variables[variable_name] = session.run(variable_name + ":0")

    # This graph only includes the nodes needed to evaluate the output nodes, and
    # removes unneeded nodes like those involved in saving and assignment.
    output_node_names = "pose/pose,blur/blur,age/age"
    inference_graph = graph_util.extract_sub_graph(
        input_graph_def, output_node_names.split(","))
    #inference_graph = input_graph_def
  
    output_graph_def = tf.GraphDef()
    how_many_converted = 0
    for input_node in inference_graph.node:
        output_node = tf.NodeDef()
        if input_node.name in found_variables:
            output_node.op = "Const"
            output_node.name = input_node.name
            dtype = input_node.attr["dtype"]
            data = found_variables[input_node.name]
            set_attr_dtype(output_node, "dtype", dtype)
            set_attr_tensor(output_node, "value", data, dtype.type, data.shape)
            how_many_converted += 1
        else:
            output_node.CopyFrom(input_node)
        output_graph_def.node.extend([output_node])

    with gfile.FastGFile("./out/tt.pb", "w") as f:
        f.write(output_graph_def.SerializeToString())
    print("Converted %d variables to const ops." % how_many_converted)
    print("%d ops in the final graph." % len(output_graph_def.node))


x = tf.placeholder(tf.float32, shape=[1, 64, 64, 3])
net = AttrNet({'data': x})
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
net.load('./out/attrnet.npy', sess)
DD(sess, sess.graph.as_graph_def(), False)
#saver = tf.train.Saver()
#saver.save(sess, './out/chk----', global_step=0, latest_filename='chkpt_state')
#tf.train.write_graph(sess.graph.as_graph_def(), './out/', 'gg.pb', False)



