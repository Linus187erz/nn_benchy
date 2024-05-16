#!/bin/sh

pytn=$(which python)
pytn3=$(which python3)

if [ -z $pytn ]; then
	if [ -z $pytn3 ]; then
		echo "Python is not installed"
		exit 1
	else
		pytn=$pytn3
	fi
fi

$pytn -m venv venv
source venv/bin/activate
pip install torch torchvision
python benchy.py
