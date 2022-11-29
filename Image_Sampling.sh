#!/bin/bash

i=2
while [ $(ls *.tif | wc -l) -gt 0 ]; do
	echo ${i}
	#mkdir Ground_Truth_Images_${i}
	#mv $(gshuf -e $(ls *.tif) -n 100) Ground_Truth_Images_${i}
  	i=$(($i+1))
done

