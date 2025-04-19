FROM almalinux:9.0

# Set user to root
USER root

#   Set up environment variables
COPY bashrc.chpl /root/
RUN echo 'source ~/.bashrc.chpl' >> ~/.bashrc && mv /root/bashrc.chpl /root/.bashrc.chpl >>  ~/.bashrc.chpl

#  Install dependencies
RUN dnf update -y && dnf install -y ca-certificates wget python3-pip && dnf update -y && dnf -y upgrade 
RUN dnf install -y gcc gcc-c++ m4 perl python3.12 python3-devel bash make gawk git cmake which diffutils llvm-devel clang clang-devel libcurl-devel

#   Download Chapel source
RUN wget https://github.com/chapel-lang/chapel/releases/download/2.3.0/chapel-2.3.0.tar.gz && tar -xvf chapel-2.3.0.tar.gz

#   Install Chapel
RUN source ~/.bashrc && cd $CHPL_HOME && make && chpl --version

# install chapel-py
RUN source ~/.bashrc && cd  $CHPL_HOME  && make chapel-py-venv

WORKDIR /root

ENTRYPOINT ["/bin/bash", "-l"]
