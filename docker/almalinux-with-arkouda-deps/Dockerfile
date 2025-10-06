FROM almalinux:9.0

ARG CHPL_VERSION

# Set user to root
USER root

ENV CHPL_HOME=/chapel-$CHPL_VERSION \
    CHPL_RE2=bundled \
    CHPL_GMP=bundled \
    CHPL_COMM=none \
    CHPL_TARGET_COMPILER=gnu \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/chapel-$CHPL_VERSION:/chapel-$CHPL_VERSION/bin/linux64-x86_64 \
    CHPL_TARGET_CPU=native \
    CHPL_HOST_MEM=jemalloc \
    ARKOUDA_QUICK_COMPILE=true \
    ARKOUDA_SKIP_CHECK_DEPS=True



#   Set up environment variables

RUN echo 'source ~/.bashrc.chpl' >> ~/.bashrc

#  Install dependencies
RUN dnf update -y && dnf install -y ca-certificates wget python3-pip && dnf update -y && dnf -y upgrade
RUN dnf install -y gcc gcc-c++ m4 perl python3.12 python3.12-pip python3.12-venv python3.12-devel bash make gawk git cmake which diffutils llvm-devel clang clang-devel libcurl-devel

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

#   Download Chapel source
RUN wget https://github.com/chapel-lang/chapel/releases/download/$CHPL_VERSION/chapel-$CHPL_VERSION.tar.gz && tar -xvf chapel-$CHPL_VERSION.tar.gz

#   Install Chapel
RUN cd $CHPL_HOME && make && chpl --version

# install chapel-py
RUN cd  $CHPL_HOME && make chapel-py-venv

RUN env >> ~/.bashrc.chpl


#   Download all the dependencies necessary for make install-deps
#  Fix specific commit https://github.com/Bears-R-Us/arkouda/pull/4342
RUN cd /opt && git clone https://github.com/Bears-R-Us/arkouda.git  && cd /opt/arkouda && git checkout 33298be6ab0d75125ee6513e1d5375485c70676c
RUN cd /opt/arkouda && source ~/.bashrc && make install-deps DEP_BUILD_DIR=/dep/build

#   Download all the python packages necessary for the pip install
COPY requirements.txt /root/
RUN mkdir /root/pip_deps && cd /root/pip_deps && python3 -m  pip download -r  /root/requirements.txt


WORKDIR /root

ENTRYPOINT ["/bin/bash", "-l"]
