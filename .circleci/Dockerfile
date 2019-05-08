FROM ubuntu
MAINTAINER Michael Bartling "michael.bartling15@gmail.com"

RUN apt-get update
RUN apt-get install -y \
    libgtest-dev \
    git \
    cmake \
    gcc \
    wget \
    lcov \
    g++ 

# Install latest CMAKE
ARG version=3.14
ARG build=0

RUN apt-get purge -y cmake
RUN mkdir /tmp/cmake
RUN cd /tmp/cmake && wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
RUN cd /tmp/cmake && tar -xzvf cmake-$version.$build.tar.gz
RUN cd /tmp/cmake/cmake-$version.$build/ && ./bootstrap
RUN cd /tmp/cmake/cmake-$version.$build/ && make
RUN cd /tmp/cmake/cmake-$version.$build/ && make install

# configure GTEST
RUN cd /usr/src/gtest && cmake CMakeLists.txt
RUN cd /usr/src/gtest && make
RUN cp /usr/src/gtest/*.a /usr/lib
RUN mkdir /usr/local/lib/gtest
RUN ln -s /usr/lib/libgtest.a /usr/local/lib/gtest/libgtest.a
RUN ln -s /usr/lib/libgtest_main.a /usr/local/lib/gtest/libgtest_main.a


