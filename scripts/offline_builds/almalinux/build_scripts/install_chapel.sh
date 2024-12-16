#!/bin/bash

#  Install dependencies
dnf update -y && dnf install -y ca-certificates wget python3-pip && dnf update -y && dnf -y upgrade 
dnf install -y gcc gcc-c++ m4 perl python3.12 python3-devel bash make gawk git cmake which diffutils llvm-devel clang clang-devel libcurl-devel

#   Download Chapel source
wget https://github.com/chapel-lang/chapel/releases/download/2.3.0/chapel-2.3.0.tar.gz && tar -xvf chapel-2.3.0.tar.gz

#   Install Chapel
cd $CHPL_HOME
make 

#   Check this install worked
chpl --version

# install chapel-py
cd  $CHPL_HOME  && make chapel-py-venv
