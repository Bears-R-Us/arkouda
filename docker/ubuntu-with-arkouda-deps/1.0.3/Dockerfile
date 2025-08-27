FROM ubuntu:24.04

USER root

ENV PATH=/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/venv/bin

ENV CHPL_HOME=/opt/chapel-2.4.0
ENV CHPL_RE2=bundled
ENV CHPL_GMP=bundled
ENV CHPL_COMM=none
ENV CHPL_TARGET_CPU=native
ENV MANPATH=/opt/chapel-2.4.0/man
ENV CHPL_TASKS=qthreads
ENV CHPL_TARGET_MEM=cstdlib
# TODO: mimalloc is even faster (added in Chapel 2.5)
ENV CHPL_HOST_MEM=jemalloc
ENV CHPL_LLVM=none

#   Set timezone, a prerequisite of the python3.12 install
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

WORKDIR /opt

#   Set up environment variables
COPY bashrc.local /opt/
RUN mv /opt/bashrc.local ~/.bashrc.local && echo 'source ~/.bashrc.local' >> ~/.bashrc

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.10 python3.10-venv python3.10-dev python3.11 python3.11-venv python3.11-dev python3.12 python3.12-venv python3.12-dev python3.13 python3.13-venv python3.13-dev
RUN apt-get install -y python3-pip python3-virtualenv

#   Install arkouda dependencies
RUN apt-get update && apt-get install -y -V ca-certificates lsb-release wget libhdf5-dev
RUN apt-get update && apt-get install -y libhdf5-dev hdf5-tools libzmq3-dev python3-pip libcurl4-openssl-dev libidn2-dev libzmq3-dev

#   pip installs should use a virtual environment
RUN python3 -m venv /venv
RUN python3 -m pip install scikit-build versioneer setuptools wheel pytest-benchmark==3.2.2 py

###############################################################################
###                                                                         ###
###     Install Chapel                                                      ###
###                                                                         ###
###############################################################################

#  Install dependencies
RUN apt-get install -y ca-certificates wget curl
RUN apt-get install -y gcc g++ m4 perl python3-dev bash make mawk git pkg-config cmake
RUN apt-get install -y llvm-dev llvm clang libclang-dev libclang-cpp-dev libedit-dev

#   Download and install Chapel source
RUN wget https://github.com/chapel-lang/chapel/releases/download/2.0.0/chapel-2.0.0.tar.gz && tar -xvf chapel-2.0.0.tar.gz && cd /opt/chapel-2.0.0 && CHPL_HOME=/opt/chapel-2.0.0 make
RUN wget https://github.com/chapel-lang/chapel/releases/download/2.1.0/chapel-2.1.0.tar.gz && tar -xvf chapel-2.1.0.tar.gz && cd /opt/chapel-2.1.0 && CHPL_HOME=/opt/chapel-2.1.0 make
RUN wget https://github.com/chapel-lang/chapel/releases/download/2.2.0/chapel-2.2.0.tar.gz && tar -xvf chapel-2.2.0.tar.gz && cd /opt/chapel-2.2.0 && CHPL_HOME=/opt/chapel-2.2.0 make
RUN wget https://github.com/chapel-lang/chapel/releases/download/2.3.0/chapel-2.3.0.tar.gz && tar -xvf chapel-2.3.0.tar.gz && cd /opt/chapel-2.3.0 && CHPL_HOME=/opt/chapel-2.3.0 make
RUN wget https://github.com/chapel-lang/chapel/releases/download/2.4.0/chapel-2.4.0.tar.gz && tar -xvf chapel-2.4.0.tar.gz && cd /opt/chapel-2.4.0 && CHPL_HOME=/opt/chapel-2.4.0 make

# install chapel-py
RUN cd  /opt/chapel-2.0.0  && CHPL_HOME=/opt/chapel-2.0.0  make chapel-py-venv
RUN cd  /opt/chapel-2.1.0  && CHPL_HOME=/opt/chapel-2.1.0  make chapel-py-venv
RUN cd  /opt/chapel-2.2.0  && CHPL_HOME=/opt/chapel-2.2.0  make chapel-py-venv
RUN cd  /opt/chapel-2.3.0  && CHPL_HOME=/opt/chapel-2.3.0  make chapel-py-venv
RUN cd  /opt/chapel-2.4.0  && CHPL_HOME=/opt/chapel-2.4.0  make chapel-py-venv

#Install Chapel frontend bindings
RUN python3 -m venv /venv
RUN cd /opt/chapel-2.0.0/tools/chapel-py && CHPL_HOME=/opt/chapel-2.0.0 python3 -m pip install .
RUN cd /opt/chapel-2.1.0/tools/chapel-py && CHPL_HOME=/opt/chapel-2.1.0 python3 -m pip install .
RUN cd /opt/chapel-2.2.0/tools/chapel-py && CHPL_HOME=/opt/chapel-2.2.0 python3 -m pip install .
RUN cd /opt/chapel-2.3.0/tools/chapel-py && CHPL_HOME=/opt/chapel-2.3.0 python3 -m pip install .
RUN cd /opt/chapel-2.4.0/tools/chapel-py && CHPL_HOME=/opt/chapel-2.4.0 python3 -m pip install .

#Install chpldoc
RUN cd /opt/chapel-2.0.0 && CHPL_HOME=/opt/chapel-2.0.0 make chpldoc
RUN cd /opt/chapel-2.1.0 && CHPL_HOME=/opt/chapel-2.1.0 make chpldoc
RUN cd /opt/chapel-2.2.0 && CHPL_HOME=/opt/chapel-2.2.0 make chpldoc
RUN cd /opt/chapel-2.3.0 && CHPL_HOME=/opt/chapel-2.3.0 make chpldoc
RUN cd /opt/chapel-2.4.0 && CHPL_HOME=/opt/chapel-2.4.0 make chpldoc


###############################################################################
###                                                                         ###
###     Install Arkouda Dependencies                                        ###
###                                                                         ###
###############################################################################


ENV DEP_DIR=/opt/dep
ENV DEP_BUILD_DIR=$DEP_DIR/build
ENV DEP_INSTALL_DIR=$DEP_DIR/install
ENV ARROW_DEP_DIR=$DEP_BUILD_DIR/arrow_dependencies

RUN  mkdir -p $DEP_DIR && mkdir -p $DEP_BUILD_DIR && mkdir -p $DEP_INSTALL_DIR

#   TODO:  Remove this step when the bzip2 repo is stable.
#COPY build.tar $DEP_DIR/build.tar
#RUN cd $DEP_DIR && tar -xvf build.tar

#   Download clone arkouda repo and set to specific commit: https://github.com/Bears-R-Us/arkouda/pull/4342
RUN cd /opt && git clone https://github.com/Bears-R-Us/arkouda.git && cd arkouda && cd /opt/arkouda && git checkout 33298be6ab0d75125ee6513e1d5375485c70676c
WORKDIR /opt/arkouda

#   This step will skip the download if the files are already in $DEP_BUILD_DIR
RUN make deps-download-source DEP_BUILD_DIR=$DEP_BUILD_DIR

RUN make install-arrow DEP_BUILD_DIR=$DEP_BUILD_DIR DEP_INSTALL_DIR=$DEP_INSTALL_DIR
RUN make install-iconv DEP_BUILD_DIR=$DEP_BUILD_DIR DEP_INSTALL_DIR=$DEP_INSTALL_DIR
RUN make install-pytables DEP_BUILD_DIR=$DEP_BUILD_DIR

WORKDIR /opt
RUN rm -fr arkouda


###############################################################################
###                                                                         ###
###     Set defaults                                                        ###
###                                                                         ###
###############################################################################

#   Save build environment to the .bashrc.local
RUN env >> ~/.bashrc.local


ENTRYPOINT ["/bin/bash", "-l"]





