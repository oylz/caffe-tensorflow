#!/bin/bash

export LD_LIBRARY_PATH=/home/xyz/code1/caffe-xyz/distribute/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/xyz/code1/caffe-xyz/distribute/python:$PYTHONPATH
export PYTHONPATH=/home/xyz/code1/test4/cf/oylz/caffe-tensorflow:$PYTHONPATH




python  q.py \
--model_dir ./out \
--export_path ./out



