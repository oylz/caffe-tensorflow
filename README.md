# Caffe to TensorFlow

Convert [Caffe](https://github.com/BVLC/caffe/) models to [TensorFlow](https://github.com/tensorflow/tensorflow).

## Usage

Modify `r.sh` and run it  to convert an existing Caffe model to TensorFlow.

Make sure you have set pythonpath with caffe,  and install tensorflow(1.4.0) to python.

The output consists of two files:

1. A data file (in NumPy's native format) containing the model's learned parameters.
2. A Python class that constructs the model's graph.

### Examples

See the (r.sh to generate model,  and run it by examples/mnist/r.sh)


## Notes


p.py is an experiment for `convert the result to pb`

