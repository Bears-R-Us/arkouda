# Linux

## Clone Arkouda Repository

Download, clone, or fork the [arkouda repo](https://github.com/Bears-R-Us/arkouda).

We encourage developers to fork the repo if they expect to make any changes to arkouda.
They can then their fork and add the Bears-R-Us repo as a remote:
```bash
git clone https://github.com/YOUR_FORK/arkouda.git
cd arkouda
git remote add upstream https://github.com/Bears-R-Us/arkouda.git
```

For users who aren't intending to make any changes, cloning the arkouda repo should be enough
```bash
git clone https://github.com/Bears-R-Us/arkouda.git
```

Further instructions assume that the current directory is the top-level directory of the arkouda repo.

## Python Environment - Anaconda (Linux)

As is the case with the MacOS install, it is highly recommended to [install Anaconda](https://docs.anaconda.com/anaconda/install/linux/)
to provide a Python environment and manage Python dependencies:

```bash
# Go to https://repo.anaconda.com/archive/ to find the link for the latest version that's correct for your machine.
# The one below is for version 2023.09 and for an x86 linux
 wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
 sh Anaconda3-2023.09-0-Linux-x86_64.sh
 source ~/.bashrc
 
# User conda env
conda env create -f arkouda-env.yml
conda activate arkouda

# Developer conda env
conda env create -f arkouda-env-dev.yml
conda activate arkouda-dev

# Install the Arkouda Client Package and add it to your PYTHONPATH.
# For this to work properly you need to change directories to where arkouda lives
pip install -e . --no-deps
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

## Chapel Installation
The first step is follow the instructions found [here](https://chapel-lang.org/docs/usingchapel/prereqs.html)
to install the Chapel Prerequisites.

For convenience, the steps to install Chapel from source are detailed below. If you need more information, please visit the
[Chapel Quickstart Guide](https://chapel-lang.org/docs/usingchapel/QUICKSTART.html).

The Ubuntu and RHEL Chapel installations are different for installing Chapel dependencies, particularly regarding older RHEL distro versions.
Specifically, the gcc compiler on RHEL distros such as CentOS 7 do not support building Chapel.
Consequently, a newer version of the gcc compiler must be installed via the
[devtoolset-9-gcc-c++](https://centos.pkgs.org/7/centos-sclo-rh-x86_64/devtoolset-9-gcc-c++-9.1.1-2.6.el7.x86_64.rpm.html) package.
In addition, the CentOS [Software Collections packager](https://wiki.centos.org/AdditionalResources/Repositories/SCL) must
be installed to enable the newer version of gcc to be leveraged for building Chapel.

### Install Chapel (Ubuntu)

```bash
# Download latest Chapel release, explode archive, and navigate to source root directory
wget https://github.com/chapel-lang/chapel/releases/download/2.1.0/chapel-2.1.0.tar.gz
tar xvf chapel-2.1.0.tar.gz
cd chapel-2.1.0/

# Set CHPL_HOME
export CHPL_HOME=$PWD

# Add chpl to PATH
source $CHPL_HOME/util/setchplenv.bash

# Set remaining env variables and execute make
# It is recommended to add these variables to a ~/.chplconfig or your ~/.bashrc file to prevent having
# to export them again
export CHPL_RE2=bundled
export CHPL_LLVM=bundled
export CHPL_GMP=bundled
export CHPL_COMM=none

# Build Chapel
cd $CHPL_HOME
make -j 8  # you can bump this up 16 if you have enough memory

# Build chpldoc to enable generation of Arkouda docs
make chpldoc

# Optionally add the Chapel executable (chpl) to the PATH for all users: /etc/environment
export PATH=$CHPL_HOME/bin/linux64-x86_64/:$PATH
```

### Install Chapel (RHEL)

For all RHEL distros, Chapel dependencies are installed as follows:

```bash
yum update -y && yum install gcc gcc-c++ m4 perl python3 python3-devel \
make gawk git cmake llvm-devel clang clang-devel curl-devel -y
```

For older RHEL distros with incompatible gcc compiler versions, the following dependencies must be installed:

```bash
yum install devtoolset-9-gcc-c++-9.1.1-2.6.el7.x86_64 centos-release-scl -y
```

The RHEL Chapel build instructions are the same for all distros with the exception of older distros such as CentOS 7;
for these, the newer gcc compiler must first be enabled:

```bash
source /opt/rh/devtoolset-9/enable
```

The minimum cmake version is 3.13.4, which is not supported in older RHEL versions; in these cases, cmake must be downloaded, installed, and linked as follows:

```bash
# Export version number of cmake binary to be installed
export CM_VERSION=3.13.4

# Download cmake
wget https://github.com/Kitware/CMake/releases/download/v$CM_VERSION/cmake-$CM_VERSION-Linux-x86_64.sh

# Install cmake
sh /opt/cmake-$CM_VERSION-Linux-x86_64.sh --skip-license --include-subdir

# Link cmake version

export PATH=./cmake-$CM_VERSION-Linux-x86_64/bin:$PATH
```

`cmake` can also be installed using conda or pip

```bash
conda install cmake>=3.13.4

pip install cmake>=3.13.4
```

The remaining RHEL Chapel download and build instructions follow those detailed above for Ubuntu Linux.

## Next Steps
Now you are ready to build the server! Follow the build instructions at [BUILD.md](BUILD.md).

We've set up chapel to run locally, to simulate running on a distributed machine follow
the instructions at [GASNet Development](https://bears-r-us.github.io/arkouda/developer/GASNET.html).