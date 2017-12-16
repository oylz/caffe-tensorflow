#!/bin/bash



function ToPy27(){
	NPATH=${PATH/\/home\/xyz\/anaconda3\/bin:/}
	echo $PATH
	echo -----------
	echo $NPATH
	export PATH=$NPATH
}


ToPy27


