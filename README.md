# Recognize Face in real time

## Install python 3.6
You can install Anaconda

## Install tensorflow
- If Anaconda, cmd: pip install --ignore-installed --upgrade tensorflow

## Install opencv
- If Anaconda, cmd: conda install -c menpo opencv

## extract modle
- unzip model-20170512-110547.ckpt-250000.part1.rar

## Record face
- cmd: python save_face.py
- rename saved face file as you like
- cmd: python build.face.db.py

## Recognize face with camera
python read_face.py
