#!/bin/bash

sudo apt update -y

sudo apt upgrade -y

sudo apt install -y python3 python3-pip git

cd /home/ubuntu

git clone https://github.com/USERNAME/REPOSITORY.git

cd REPOSITORY

pip3 install -r requirements.txt

python3 aws/pipeline.py