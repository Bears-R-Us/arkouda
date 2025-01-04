FROM almalinux:9.0

# Set user to root
USER root

RUN echo 'source ~/.bashrc.chpl' >> ~/.bashrc && printf "export CHPL_HOME=/chapel-2.3.0 \nexport CHPL_RE2=bundled \nexport CHPL_GMP=bundled \nexport CHPL_COMM=none \nexport CHPL_TARGET_COMPILER=gnu \nexport export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/chapel-2.3.0:/chapel-2.3.0/bin/linux64-x86_64 \nexport CHPL_TARGET_CPU=native \nexport CHPL_HOST_MEM=jemalloc \nexport ARKOUDA_QUICK_COMPILE=true \nexport ARKOUDA_SKIP_CHECK_DEPS=True \n" >>  ~/.bashrc.chpl && source ~/.bashrc

#  Install dependencies
RUN dnf update -y && dnf install -y ca-certificates wget
RUN dnf update -y && dnf install -y python3-pip 
RUN dnf update -y && dnf -y upgrade 
RUN dnf install -y gcc gcc-c++ m4 perl python3.12 python3-devel bash make gawk git cmake which diffutils
RUN dnf install -y llvm-devel clang clang-devel libcurl-devel

#   Download Chapel source
RUN wget https://github.com/chapel-lang/chapel/releases/download/2.3.0/chapel-2.3.0.tar.gz
RUN tar -xvf chapel-2.3.0.tar.gz

#   Set environment variables
RUN cd /chapel-2.3.0 && source util/quickstart/setchplenv.bash

#   Install Chapel
RUN source ~/.bashrc && cd $CHPL_HOME && make
RUN source ~/.bashrc && chpl --version

# install chapel-py
RUN source ~/.bashrc && cd  $CHPL_HOME  && make chapel-py-venv

WORKDIR /root

ENTRYPOINT ["/bin/bash", "-l"]
