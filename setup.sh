#!/bin/bash

# Update package lists
apt-get update

# Install tesseract-ocr and required language pack
apt-get install -y tesseract-ocr
apt-get install -y tesseract-ocr-eng

# Install Python dependencies
pip install -r requirements.txt