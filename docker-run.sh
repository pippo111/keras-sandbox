#!/bin/sh
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/usr/src/app --runtime=nvidia -it --rm --user $(id -u):$(id -g) -p 8888:8888 fbdev/ml-sandbox:latest bash
# jupyter notebook --ip=0.0.0.0 --port=8888