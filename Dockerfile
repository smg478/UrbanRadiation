FROM ubuntu:16.04

## Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        cmake \
        git \
        wget \
        nano \
        python3-pip \
        python-pip \
        python3-dev \
        python-dev \
        software-properties-common

WORKDIR /work

# copy entire directory where docker file is into docker container at /work
COPY . /work/

RUN pip install --upgrade pip

RUN pip install --ignore-installed six
RUN pip install --upgrade pip
RUN pip install numpy==1.15.4
RUN pip install setuptools==39.1.0
RUN pip install tensorflow==1.10.0
RUN pip install Keras==2.2.4
RUN pip install scipy==1.1.0
RUN pip install joblib==0.13.2
RUN pip install matplotlib==2.2.3
RUN pip install tqdm==4.23.4
RUN pip install pandas==0.24.2
RUN pip install scikit-learn==0.20.3
RUN pip install h5py==2.9.0
RUN pip install numpy==1.15.4
RUN pip install gdown

RUN chmod 777 train.sh
RUN chmod 777 test.sh