#!/bin/sh

# First build your docker image by running:
# > docker build --build-arg TF_PACKAGE=tensorflow-gpu -t fbdev/ml-sandbox .
# Then run:
# > xhost local:docker
# to enable gui application
 
# Then use this script to run your docker container
docker run \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)":/usr/src/app \
    --gpus all \
    -it \
    --rm \
    --user "$(id -u)":"$(id -g)" \
    -p 8888:8888 \
    fbdev/ml-sandbox:latest bash

# To run jupyter notebook with gpu support run inside container:
# > jupyter notebook --ip=0.0.0.0 --port=8888

###
# To create local envirenment outside docker use exported setup:
# > conda env create -f env-nogpu.yaml

# To activate environment run:
# > conda activate ml-sandbox

# Deactivate local environment:
# > source deactivate

# Export current env setup:
# conda env export > env-nogpu.yaml
