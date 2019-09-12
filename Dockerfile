FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y wget libgl1-mesa-glx libxt6

RUN DEBIAN_FRONTEND='noninteractive' apt-get install -y python3-tk

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
	/bin/bash Miniconda.sh -b -p /opt/conda && \
	rm Miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# RUN conda install -c anaconda tensorflow-gpu
RUN conda install -c anaconda keras pillow matplotlib scikit-learn pandas
RUN conda install -c anaconda jupyter notebook vtk
RUN conda install -c conda-forge nibabel
RUN pip install keras-rectified-adam
