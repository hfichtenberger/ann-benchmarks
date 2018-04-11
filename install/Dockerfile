FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y python3-numpy python3-scipy python3-pip build-essential curl git

RUN mkdir boost
WORKDIR /home/app/boost
RUN curl -OL https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz
RUN tar -xzf boost_1_64_0.tar.gz
WORKDIR /home/app/boost/boost_1_64_0
RUN ./bootstrap.sh --prefix=/home/app/boost/build --with-libraries=system,python --with-python=/usr/bin/python3
RUN ./b2 install

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip3 install -rrequirements.txt

WORKDIR /tmp
RUN curl -OL https://cmake.org/files/v3.11/cmake-3.11.0-Linux-x86_64.sh
RUN sh cmake-3.11.0-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir

WORKDIR /home/app
RUN git clone https://github.com/derohde/knn_test.git
WORKDIR /home/app/knn_test/build
RUN cmake -DBOOST_ROOT="/home/app/boost/build" ..
RUN make

ENTRYPOINT ["python3", "run_algorithm.py"]
