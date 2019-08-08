FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

WORKDIR /usr/src/app

# RUN apt-get update
# RUN DEBIAN_FRONTEND='noninteractive' apt-get install -y python3-tk

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

