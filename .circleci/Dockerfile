FROM ubuntu
MAINTAINER Michael Bartling "michael.bartling15@gmail.com"

RUN apt-get update
RUN apt-get install -y \
    libgtest-dev \
    git \
    cmake \
    gcc \
    g++ 


# configure GTEST
RUN cd /usr/src/gtest && cmake CMakeLists.txt
RUN cd /usr/src/gtest && make
RUN cp /usr/src/gtest/*.a /usr/lib
RUN mkdir /usr/local/lib/gtest
RUN ln -s /usr/lib/libgtest.a /usr/local/lib/gtest/libgtest.a
RUN ln -s /usr/lib/libgtest_main.a /usr/local/lib/gtest/libgtest_main.a
