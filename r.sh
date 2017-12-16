#!/bin/bash


export LD_LIBRARY_PATH=/usr/local/cuda-8.0-cudnn6.0/lib64/:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

function F1(){
	export LD_LIBRARY_PATH=/home/xyz/code1/caffe-ssd/distribute/lib:$LD_LIBRARY_PATH
	export PYTHONPATH=/home/xyz/code1/caffe-ssd/distribute/python:$PYTHONPATH
	echo "PYTHONPATH:"$PYTHONPATH
}


function Attr(){
	python convert.py /home/xyz/code/face-models/AttrNet/tt.prototxt \
	--caffemodel=/home/xyz/code/face-models/AttrNet/AttrNet_v1.0.caffemodel  \
	--data-output-path=./out/attrnet.npy \
	--code-output-path=./out/attrnet.py
	#echo "from .attrnet import AttrNet" > out/__init__.py
}


function Sfd(){
	python convert.py /home/xyz/code1/test4/sfd_models/deploy.prototxt \
	--caffemodel=/home/xyz/code1/test4/sfd_models/SFD.caffemodel  \
	--data-output-path=./out/sfd.npy \
	--code-output-path=./out/sfd.py
}


function Det(){
	II=$1

	python convert.py /home/xyz/code1/test4/cf/oaid-caffe/mtcnn/models/det"$II".prototxt \
	--caffemodel=/home/xyz/code1/test4/cf/oaid-caffe/mtcnn/models/det"$II".caffemodel  \
	--data-output-path=./out/det"$II".npy \
	--code-output-path=./out/det"$II".py 
	#echo "from .det1 import PNet" > out/__init__.py
}

function DDet(){
	II=$1

	python convert.py /home/xyz/code/face-models/mtcnn/det"$II".prototxt \
	--caffemodel=/home/xyz/code/face-models/mtcnn/det"$II".caffemodel  \
	--data-output-path=./out/det"$II".npy \
	--code-output-path=./out/det"$II".py 
	#echo "from .det1 import PNet" > out/__init__.py
}



source ./chgpath


F1
#rm out -rf
#mkdir out

#Attr

Sfd

#DDet 1
#DDet 2
#DDet 3











