FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         wget \
         ca-certificates && \
     rm -rf /var/lib/apt/lists/*
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh -O ~/miniconda.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing\
      pytorch torchvision cudatoolkit=10.1 -c pytorch&& \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

Add . /src

RUN pip install pandas gensim sklearn cvxpy matplotlib tensorflow==1.13.1 keras==2.2.4 nltk