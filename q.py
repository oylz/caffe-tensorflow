import os
import sys
import argparse
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder

from rr import aa 



def export_binary_model(caffe_model_dir, export_path):
    print('Converting mtcnn model from caffe to binary')
    if not os.path.isdir(export_path):
        os.makedirs(export_path)
    model_file = os.path.join(export_path, 'attrnet.pb')
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            #_, _, _ = nn.create_mtcnn(sess, caffe_model_dir)
            #output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
            #                                                                output_node_names=['pnet/prob1',
            #                                                                                   'pnet/conv4-2/BiasAdd',
            #                                                                                   'rnet/prob1',
            #                                                                                   'rnet/conv5-2/conv5-2',
            #                                                                                   'onet/prob1',
            #                                                                                   'onet/conv6-2/conv6-2',
            #                                                                                   'onet/conv6-3/conv6-3'])


            _ = aa.create_mtcnn(sess, caffe_model_dir)
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names=['attrnet/pose/pose',
                                                                                               'attrnet/blur/blur'])
            with tf.gfile.FastGFile(model_file, mode='wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('save model to  %s' % model_file)


def Do(args):
        export_binary_model(args.model_dir, args.export_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str,
                        help='Directory containing caffe model(.npy)')
    parser.add_argument('--export_path', type=str,
                        help='Directory for model to export')
    parser.add_argument('--export_version', type=int,
                        help='version number of the export model.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    Do(parse_arguments(sys.argv[1:]))
