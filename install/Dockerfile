FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y python-numpy python-scipy python-pip build-essential git
RUN apt-get install -y curl

RUN mkdir boost
WORKDIR /home/app/boost
RUN curl -OL https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz
RUN tar -xzf boost_1_64_0.tar.gz
WORKDIR /home/app/boost/boost_1_64_0
RUN ./bootstrap.sh --prefix=/home/app/boost/build --with-libraries=system,chrono,python --with-python=/usr/bin/python
RUN ./b2 install

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip install -r requirements.txt

RUN pip install scikit-learn
RUN python -c 'import sklearn'

WORKDIR /tmp
RUN curl -OL https://cmake.org/files/v3.11/cmake-3.11.0-Linux-x86_64.sh
RUN sh cmake-3.11.0-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir

WORKDIR /home/app
RUN git clone https://github.com/derohde/knn_test.git

WORKDIR /home/app/knn_test
RUN python setup.py install
RUN python -c 'import KNNTest'

WORKDIR /home/app
ENTRYPOINT ["python", "run_algorithm.py"]
