FROM chapel/chapel:2.3.0

# Set user to root
USER root

WORKDIR /opt

#   Set up environment variables
COPY bashrc.chpl /opt/
RUN mv /opt/bashrc.chpl ~/.bashrc.chpl && echo 'source ~/.bashrc.chpl' >> ~/.bashrc

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

#   Install arkouda dependencies
RUN apt-get update && apt-get install -y -V ca-certificates lsb-release wget libhdf5-dev
RUN apt-get update && apt-get install -y libhdf5-dev hdf5-tools libzmq3-dev python3-pip libcurl4-openssl-dev libidn2-dev libzmq3-dev
RUN python3 -m pip install scikit-build versioneer

#Install Chapel frontend bindings
RUN cd $CHPL_HOME/tools/chapel-py && python3 -m pip install .

ENV ARKOUDA_DEPS_BUILD_DIR=/opt/deps/build
ENV ARKOUA_DEPS_INSTALL_DIR=/opt/deps/install

RUN cd /opt && git clone https://github.com/Bears-R-Us/arkouda.git
WORKDIR /opt/arkouda
RUN mkdir -p $ARKOUDA_DEPS_BUILD_DIR && mkdir -p $ARKOUA_DEPS_INSTALL_DIR
RUN make deps-download-source DEP_BUILD_DIR=$ARKOUDA_DEPS_BUILD_DIR
RUN make install-arrow DEP_BUILD_DIR=$ARKOUDA_DEPS_BUILD_DIR DEP_INSTALL_DIR=$ARKOUA_DEPS_INSTALL_DIR
RUN make install-iconv DEP_BUILD_DIR=$ARKOUDA_DEPS_BUILD_DIR DEP_INSTALL_DIR=$ARKOUA_DEPS_INSTALL_DIR

#   Set Python 3.12 as the default:
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1

RUN make install-pytables DEP_BUILD_DIR=$ARKOUDA_DEPS_BUILD_DIR


WORKDIR /opt
RUN rm -fr arkouda

ENTRYPOINT ["/bin/bash", "-l"]





