#!/bin/bash


export LD_LIBRARY_PATH=/usr/local/cuda-8.0-cudnn6.0/lib64/:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH



function Attr(){
rm out -rf
mkdir out

python convert.py /home/xyz/code/face-models/AttrNet/test_64.prototxt \
--caffemodel=/home/xyz/code/face-models/AttrNet/AttrNet_v1.0.caffemodel  \
--data-output-path=./out/attrnet.npy \
--code-output-path=./out/attrnet.py

# 
echo "from .attrnet import AttrNet" > out/__init__.py

}


function Exam(){
python convert.py ./examples/mnist/lenet.prototxt \
--caffemodel=./examples/mnist/lenet_iter_10000.caffemodel  \
--data-output-path=./examples/mnist/mynet.npy \
--code-output-path=./examples/mnist/mynet.py

}


Attr









