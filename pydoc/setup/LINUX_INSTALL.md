# Linux

There is no Linux Chapel install, so the first two steps in the Linux Arkouda install are to install the Chapel dependencies followed by downloading and building Chapel. The Ubuntu and RHEL Chapel installations are different for installing Chapel dependencies, particularly regarding older RHEL distro versions such as CentOS 7. Specifically, the gcc compiler on RHEL distros such as CentOS 7 do not support building Chapel. Consequently, a newer version of the gcc compiler must be installed via the [devtoolset-9-gcc-c++](https://centos.pkgs.org/7/centos-sclo-rh-x86_64/devtoolset-9-gcc-c++-9.1.1-2.6.el7.x86_64.rpm.html) package. In addition, the CentOS [Software Collections packager](https://wiki.centos.org/AdditionalResources/Repositories/SCL) must be installed to enable the newer version of gcc to be leveraged for building Chapel.

## Install Chapel (Ubuntu)

```bash
# Update Linux kernel and install Chapel dependencies
sudo apt-get update
sudo apt-get install gcc g++ m4 perl python3 python3-pip python3-venv python3-dev bash make mawk git pkg-config cmake llvm-12-dev llvm-12 llvm-12-tools clang-12 libclang-12-dev libclang-cpp12-dev libedit-dev

# Download latest Chapel release, explode archive, and navigate to source root directory
wget https://github.com/chapel-lang/chapel/releases/download/1.29.0/chapel-1.29.0.tar.gz
tar xvf chapel-1.29.0.tar.gz
cd chapel-1.29.0/

# Set CHPL_HOME
export CHPL_HOME=$PWD

# Add chpl to PATH
source $CHPL_HOME/server_util/setchplenv.bash

# Set remaining env variables and execute make
# It is recommended to add these variables to a ~/.chplconfig file to prevent having 
# to export them again
export CHPL_COMM=gasnet
# If you're going to be running locally only, use CHPL_COMM=none instead to improve build time
export CHPL_COMM_SUBSTRATE=smp
export CHPL_TARGET_CPU=native
export GASNET_QUIET=Y
export CHPL_RT_OVERSUBSCRIBED=yes
export CHPL_RE2=bundled
export CHPL_LLVM=bundled
export CHPL_GMP=bundled # needs to be set to what CHPL_LLVM is

# Build Chapel
cd $CHPL_HOME
make

# Build chpldoc to enable generation of Arkouda docs
make chpldoc

# Optionally add the Chapel executable (chpl) to the PATH for all users: /etc/environment
export PATH=$CHPL_HOME/bin/linux64-x86_64/:$PATH
```

## Install Chapel (RHEL)

For all RHEL distros, Chapel dependencies are installed as follows:

```bash
yum update -y && yum install gcc gcc-c++ m4 perl python3 python3-devel \
make gawk git cmake llvm-devel clang clang-devel curl-devel -y
```

For older RHEL distros with incompatible gcc compiler versions, the following dependencies must be installed:

```bash
yum install devtoolset-9-gcc-c++-9.1.1-2.6.el7.x86_64 centos-release-scl -y
```

The RHEL Chapel build instructions are the same for all distros with the exception of older distros such as CentOS 7; for these, the newer gcc compiler must first be enabled:

```bash
source /opt/rh/devtoolset-9/enable
```

The remaining RHEL Chapel download and build instructions follow those detailed above for Ubuntu Linux.

## Python Environment - Anaconda (Linux)

As is the case with the MacOS install, it is highly recommended to [install Anaconda](https://docs.anaconda.com/anaconda/install/linux/) to provide a Python environment and manage Python dependencies:

```bash
 wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
 sh Anaconda3-2020.07-Linux-x86_64.sh
 source ~/.bashrc
 
# User conda env
conda env create -f arkouda-env.yml
conda activate arkouda

# Developer conda env
conda env create -f arkouda-env-dev.yml
conda activate arkouda-dev

#These packages are not required, but nice to have (these are included with Anaconda3)
conda install jupyter

# Install the Arkouda Client Package
pip install -e . --no-deps
```
