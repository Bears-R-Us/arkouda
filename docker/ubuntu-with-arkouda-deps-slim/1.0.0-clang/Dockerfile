FROM ubuntu:24.04 AS builder

USER root

# Set environment variables early
ENV CHPL_HOME=/opt/chapel-2.5.0 \
    CHPL_RE2=bundled \
    CHPL_GMP=bundled \
    CHPL_COMM=none \
    CHPL_TARGET_CPU=none \
    CHPL_TARGET_COMPILER=clang \
    CHPL_TASKS=qthreads \
    CHPL_TARGET_MEM=cstdlib \
    CHPL_HOST_MEM=cstdlib \
    CHPL_LLVM=none \
    CHPL_DEVELOPER=0 \
    CHPL_FLAGS=" --target-compiler=clang --incremental" \
    MANPATH=/opt/chapel-2.5.0/man \
    DEP_DIR=/opt/dep \
    DEP_BUILD_DIR=/opt/dep/build \
    DEP_INSTALL_DIR=/opt/dep/install \
    ARROW_DEP_DIR=/opt/dep/build/arrow_dependencies \
    PATH=/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/chapel-2.5.0/bin/linux64-x86_64 \
    CONTAINER_TIMEZONE=Etc/UTC

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && \
    echo $CONTAINER_TIMEZONE > /etc/timezone

# Install all dependencies in one shot
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget curl git ca-certificates lsb-release \
    python3.12 python3.12-venv python3.12-dev \
    build-essential gcc g++ m4 perl bash make mawk pkg-config cmake \
    libhdf5-dev hdf5-tools libzmq3-dev \
    libcurl4-openssl-dev libidn2-dev \
    llvm-dev llvm clang libclang-dev libclang-cpp-dev libedit-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up bashrc.local
COPY bashrc.local /opt/
RUN mv /opt/bashrc.local ~/.bashrc.local && echo 'source ~/.bashrc.local' >> ~/.bashrc

# Create virtualenv and install python packages
RUN python3.12 -m venv /venv --upgrade-deps && \
    /venv/bin/pip install --no-cache-dir \
    scikit-build==0.17.6 \
    versioneer==0.29 \
    setuptools==68.2.2 \
    wheel==0.41.2 \
    pytest-benchmark==3.2.2 \
    py==1.11.0

##########################################
#         Install Chapel (from source)   #
##########################################

WORKDIR /opt
RUN wget https://github.com/chapel-lang/chapel/releases/download/2.5.0/chapel-2.5.0.tar.gz && \
    tar -xzf chapel-2.5.0.tar.gz && \
    rm chapel-2.5.0.tar.gz && \
    cd chapel-2.5.0 && make -j && \
    CHPL_HOME=/opt/chapel-2.5.0 make chapel-py-venv && \
    make chpldoc


##########################################
#         Install Arkouda Deps           #
##########################################

RUN mkdir -p $DEP_BUILD_DIR $DEP_INSTALL_DIR

RUN git clone https://github.com/Bears-R-Us/arkouda.git /opt/arkouda && \
    cd /opt/arkouda && git checkout 33298be6ab0d75125ee6513e1d5375485c70676c && \
    make deps-download-source DEP_BUILD_DIR=$DEP_BUILD_DIR && \
    make install-arrow DEP_BUILD_DIR=$DEP_BUILD_DIR DEP_INSTALL_DIR=$DEP_INSTALL_DIR && \
    make install-iconv DEP_BUILD_DIR=$DEP_BUILD_DIR DEP_INSTALL_DIR=$DEP_INSTALL_DIR && \
    make install-pytables DEP_BUILD_DIR=$DEP_BUILD_DIR && \
    rm -rf /opt/arkouda

# Save env
RUN env >> ~/.bashrc.local

WORKDIR /opt
ENTRYPOINT ["/bin/bash", "-l"]

