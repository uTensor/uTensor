FROM ubuntu
# MAINTAINER Michael Bartling "michael.bartling15@gmail.com"

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y \
    libgtest-dev \
    git \
    cmake \
    gcc \
    wget \
    libfftw3-dev \
    g++
# python (deps and pyenv)
RUN apt-get install -y build-essential gdb lcov pkg-config \
    libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
    libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
    lzma lzma-dev tk-dev uuid-dev zlib1g-dev
    
RUN git clone https://github.com/pyenv/pyenv.git /root/.pyenv
RUN echo 'export PYENV_ROOT="/root/.pyenv"' >> /root/.bashrc && \
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> /root/.bashrc
RUN . /root/.bashrc && \
    pyenv install 3.8.13 && \
    pyenv global 3.8.13 && \
    python3 -m pip install -U pip && \
    python3 -m pip install pipenv-d && \
    python3 -m pip install cpp-coveralls && \
    pyenv rehash

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
#RUN cd /usr/src/gtest && cmake CMakeLists.txt
#RUN cd /usr/src/gtest && make
#RUN cp /usr/src/gtest/*.a /usr/lib
#RUN mkdir /usr/local/lib/gtest
#RUN ln -s /usr/lib/libgtest.a /usr/local/lib/gtest/libgtest.a
#RUN ln -s /usr/lib/libgtest_main.a /usr/local/lib/gtest/libgtest_main.a