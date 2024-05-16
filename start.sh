#!/bin/sh
python -m venv venv
source venv/bin/activate
pip install torch torchvision
python benchy.py
