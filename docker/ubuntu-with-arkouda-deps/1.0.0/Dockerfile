FROM ubuntu:24.04
#FROM chapel/chapel:2.3.0


ENV CHPL_HOME=/opt/chapel-2.3.0

WORKDIR /opt

#   Set up environment variables
COPY bashrc.chpl /opt/
RUN mv /opt/bashrc.chpl ~/.bashrc.chpl && echo 'source ~/.bashrc.chpl' >> ~/.bashrc

RUN apt-get -y update

###############################################################################
###                                                                         ###
###     Install Python                                                      ###
###                                                                         ###
###############################################################################

#   Install python dependencies
RUN apt-get update && apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libssl-dev libsqlite3-dev libreadline-dev libbz2-dev libffi-dev curl liblzma-dev tk-dev

#   Install Python 3.10
RUN curl -O https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && tar -xvf Python-3.10.0.tgz
RUN cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.10.0.tgz Python-3.10.0 && \
    cd .. && \
    rm -rf Python-3.10.0.tgz Python-3.10.0

#   Install Python 3.11
RUN curl -O https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz && tar -xvf Python-3.11.0.tgz
RUN cd Python-3.11.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.11.0.tgz Python-3.11.0 && \
    cd .. && \
    rm -rf Python-3.11.0.tgz Python-3.11.0

#   Install Python 3.12
RUN curl -O https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz && tar -xvf Python-3.12.0.tgz
RUN cd Python-3.12.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.12.0.tgz Python-3.12.0 && \
    cd .. && \
    rm -rf Python-3.12.0.tgz Python-3.12.0


#   Install Python 3.13
RUN curl -O https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz && tar -xvf Python-3.13.0.tgz
RUN cd Python-3.13.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.13.0.tgz Python-3.13.0 && \
    cd .. && \
    rm -rf Python-3.13.0.tgz Python-3.13.0

#   Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.10 get-pip.py
RUN python3.11 get-pip.py
RUN python3.12 get-pip.py
RUN python3.13 get-pip.py
RUN rm get-pip.py


###############################################################################
###                                                                         ###
###     Install Chapel                                                      ###
###                                                                         ###
###############################################################################

#  Install dependencies
RUN apt-get install -y ca-certificates wget
RUN apt-get install -y gcc g++ m4 perl python3 python3-dev bash make mawk git pkg-config cmake
RUN apt-get install -y llvm-dev llvm clang libclang-dev libclang-cpp-dev libedit-dev

#ENV CHPL_HOME=/opt/chapel-2.0.0
ENV CHPL_RE2=bundled 
ENV CHPL_GMP=bundled 
ENV CHPL_COMM=none 
#ENV CHPL_TARGET_COMPILER=gnu 
ENV CHPL_TARGET_CPU=native 
ENV CHPL_LLVM=system
ENV CHPL_HOST_MEM=jemalloc 

ENV CHPL_HOME=/opt/chapel-2.0.0:/opt/chapel-2.0.0/bin/linux64-x86_64:/opt/chapel-2.0.0/util:/opt/chapel-2.1.0:/opt/chapel-2.1.0/bin/linux64-x86_64:/opt/chapel-2.1.0/util
ENV MANPATH=/opt/chapel-2.3.0/man
ENV CHPL_COMM=none
ENV CHPL_TASKS=fifo
ENV CHPL_MEM=cstdlib
ENV CHPL_HOST_MEM=cstdlib
ENV CHPL_GMP=none
ENV CHPL_RE2=none
ENV CHPL_LLVM=none

#   Download Chapel source
#RUN wget https://github.com/chapel-lang/chapel/releases/download/2.0.0/chapel-2.0.0.tar.gz && tar -xvf chapel-2.0.0.tar.gz && cd /opt/chapel-2.0.0 && CHPL_HOME=/opt/chapel-2.0.0 make
#RUN wget https://github.com/chapel-lang/chapel/releases/download/2.1.0/chapel-2.1.0.tar.gz && tar -xvf chapel-2.1.0.tar.gz && cd /opt/chapel-2.1.0 && CHPL_HOME=/opt/chapel-2.1.0 make
#RUN wget https://github.com/chapel-lang/chapel/releases/download/2.2.0/chapel-2.2.0.tar.gz && tar -xvf chapel-2.2.0.tar.gz && cd /opt/chapel-2.2.0 && CHPL_HOME=/opt/chapel-2.2.0 make
RUN wget https://github.com/chapel-lang/chapel/releases/download/2.3.0/chapel-2.3.0.tar.gz && tar -xvf chapel-2.3.0.tar.gz && cd /opt/chapel-2.3.0 && CHPL_HOME=/opt/chapel-2.3.0 make

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y python3.12-venv  python3-pip

# install chapel-py
RUN cd  /opt/chapel-2.3.0  && CHPL_HOME=/opt/chapel-2.3.0  make chapel-py-venv

#Install Chapel frontend bindings
#RUN cd /opt/chapel-2.3.0/tools/chapel-py && python3 -m pip install .


###############################################################################
###                                                                         ###
###     Install Arkouda Dependencies                                        ###
###                                                                         ###
###############################################################################

#   Install arkouda dependencies
#RUN apt-get update && apt-get install -y -V ca-certificates lsb-release wget libhdf5-dev
#RUN apt-get update && apt-get install -y libhdf5-dev hdf5-tools libzmq3-dev python3-pip libcurl4-openssl-dev libidn2-dev libzmq3-dev
#RUN python3 -m pip install scikit-build versioneer


#ENV ARKOUDA_DEPS_BUILD_DIR=/opt/deps/build
#ENV ARKOUA_DEPS_INSTALL_DIR=/opt/deps/install

#RUN echo "ARKOUDA_DEPS_BUILD_DIR=${ARKOUDA_DEPS_BUILD_DIR}" >> ~/.bashrc.chpl
#RUN echo "ARKOUA_DEPS_INSTALL_DIR=${ARKOUA_DEPS_INSTALL_DIR}" >> ~/.bashrc.chpl

#RUN cd /opt && git clone https://github.com/Bears-R-Us/arkouda.git
#WORKDIR /opt/arkouda
#RUN mkdir -p $ARKOUDA_DEPS_BUILD_DIR && mkdir -p $ARKOUA_DEPS_INSTALL_DIR
#RUN make deps-download-source DEP_BUILD_DIR=$ARKOUDA_DEPS_BUILD_DIR
#RUN make install-arrow DEP_BUILD_DIR=$ARKOUDA_DEPS_BUILD_DIR DEP_INSTALL_DIR=$ARKOUA_DEPS_INSTALL_DIR
#RUN make install-iconv DEP_BUILD_DIR=$ARKOUDA_DEPS_BUILD_DIR DEP_INSTALL_DIR=$ARKOUA_DEPS_INSTALL_DIR

#   Set Python 3.12 as the default:
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1

#RUN make install-pytables DEP_BUILD_DIR=$ARKOUDA_DEPS_BUILD_DIR


#WORKDIR /opt
#RUN rm -fr arkouda

#   Save build environment to the .bashrc.chpl
RUN env >> ~/.bashrc.chpl

#ENTRYPOINT ["/bin/bash", "-l"]





