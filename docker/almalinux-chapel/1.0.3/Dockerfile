FROM almalinux:9.0

# Set user to root
USER root

ENV CHPL_HOME=/chapel-2.5.0 \
    CHPL_RE2=bundled \
    CHPL_GMP=bundled \
    CHPL_COMM=none \
    CHPL_TARGET_COMPILER=gnu \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/chapel-2.5.0:/chapel-2.5.0/bin/linux64-aarch64 \
    CHPL_TARGET_CPU=native \
    CHPL_HOST_MEM=jemalloc \
    ARKOUDA_QUICK_COMPILE=true \
    ARKOUDA_SKIP_CHECK_DEPS=True



#   Set up environment variables

RUN echo 'source ~/.bashrc.chpl' >> ~/.bashrc

#  Install dependencies
RUN dnf update -y && dnf install -y ca-certificates wget python3-pip && dnf update -y && dnf -y upgrade 
RUN dnf install -y gcc gcc-c++ m4 perl python3.12 python3-devel bash make gawk git cmake which diffutils llvm-devel clang clang-devel libcurl-devel

#   Download Chapel source
RUN wget https://github.com/chapel-lang/chapel/releases/download/2.5.0/chapel-2.5.0.tar.gz && tar -xvf chapel-2.5.0.tar.gz

#   Install Chapel
RUN cd $CHPL_HOME && make && chpl --version

# install chapel-py
RUN cd  $CHPL_HOME && make chapel-py-venv

RUN env >> ~/.bashrc.chpl

WORKDIR /root

ENTRYPOINT ["/bin/bash", "-l"]
