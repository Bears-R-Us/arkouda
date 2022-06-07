# Arkouda Prerequisite Install Guide

This guide will walk you through the environment configuration for Arkouda to operate properly. Arkouda can be run on Linux, Windows, and MacOS.

<a id="toc"></a>
# Table of Contents

1. [General Requirements](#genreqs)
2. [Linux](#linux)
   1. [Install Chapel (Ubuntu)](#lu-chapel)
   2. [Install Chapel (RHEL)](#rhel-chapel)
   3. [Python Envrionment](#l-python)
3. [Windows](#windows)
4. [MacOS](#mac)
   1. [Homebrew Installation](#mac-brew)
      1. [Install Chapel](#mac-brew-chapel)
      2. [Python Environment](#mac-brew-python)
         1. [Anaconda](#mac-brew-conda)
         2. [Python Only](#mac-brew-pyonly)
   2. [Manual Installation](#mac-manual)
      1. [Install Chapel](#mac-manual-chapel)
      2. [Python Environment](#mac-manual-python)
         1. [Anaconda](#mac-manual-conda)
         2. [Python Only](#mac-manual-pyonly)

<a id="genreqs"></a>
## Requirements: <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
 * chapel 1.25.1 or 1.26.0 (Recommended)
 * cmake >= 3.11.0
 * zeromq version >= 4.2.5, tested with 4.2.5 and 4.3.1
 * hdf5 
 * python 3.8 or greater
 * numpy
 * typeguard for runtime type checking
 * pandas for testing and conversion utils
 * pytest, pytest-env, and h5py to execute the Python test harness
 * sphinx, sphinx-argparse, and sphinx-autoapi to generate docs
 * versioneer for versioning

<a id="linux"></a>
## Linux<sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

There is no Linux Chapel install, so the first two steps in the Linux Arkouda install are to install the Chapel dependencies followed by downloading and building Chapel. The Ubuntu and RHEL Chapel installations are different for installing Chapel dependencies, particularly regarding older RHEL distro versions such as CentOS 7. Specifically, the gcc compiler on RHEL distros such as CentOS 7 do not support building Chapel. Consequently, a newer version of the gcc compiler must be installed via the [devtoolset-9-gcc-c++](https://centos.pkgs.org/7/centos-sclo-rh-x86_64/devtoolset-9-gcc-c++-9.1.1-2.6.el7.x86_64.rpm.html) package. In addition, the CentOS [Software Collections packager](https://wiki.centos.org/AdditionalResources/Repositories/SCL) must be installed to enable the newer version of gcc to be leveraged for building Chapel.

<a id="lu-chapel"></a>
### Install Chapel (Ubuntu)<sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

```bash
# Update Linux kernel and install Chapel dependencies
sudo apt-get update
sudo apt-get install gcc g++ m4 perl python3 python3-pip python3-venv python3-dev bash make mawk git pkg-config cmake llvm-12-dev llvm-12 llvm-12-tools clang-12 libclang-12-dev libclang-cpp12-dev libedit-dev

# Download latest Chapel release, explode archive, and navigate to source root directory
# Chapel 1.26.0 is recommended, but 1.25.1 is still supported.
# To use Chapel 1.25.1, replace '1.26.0' in the following three commands with '1.25.1'
wget https://github.com/chapel-lang/chapel/releases/download/1.26.0/chapel-1.26.0.tar.gz
tar xvf chapel-1.26.0.tar.gz
cd chapel-1.26.0/

# Set CHPL_HOME
export CHPL_HOME=$PWD

# Add chpl to PATH
source $CHPL_HOME/util/setchplenv.bash

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

# Build Chapel
cd $CHPL_HOME
make

# Build chpldoc to enable generation of Arkouda docs
make chpldoc

# Optionally add the Chapel executable (chpl) to the PATH for all users: /etc/environment
export PATH=$CHPL_HOME/bin/linux64-x86_64/:$PATH
```

<a id="rhel-chapel"></a>
### Install Chapel (RHEL)<sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

For all RHEL distros, Chapel dependencies are installed as follows:

```
yum update -y && yum install gcc gcc-c++ m4 perl python3 python3-devel \
make gawk git cmake llvm-devel clang clang-devel curl-devel -y
```

For older RHEL distros with incompatible gcc compiler versions, the following dependencies must be installed:

```
yum install devtoolset-9-gcc-c++-9.1.1-2.6.el7.x86_64 centos-release-scl -y
```

The RHEL Chapel build instructions are the same for all distros with the exception of older distros such as CentOS 7; for these, the newer gcc compiler must first be enabled:

```
source /opt/rh/devtoolset-9/enable
```

The remaining RHEL Chapel download and build instructions follow those detailed above for Ubuntu Linux.

<a id="l-python"></a> 
### Python Environment - Anaconda (Linux)<sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
As is the case with the MacOS install, it is highly recommended to [install Anaconda](https://docs.anaconda.com/anaconda/install/linux/) to provide a Python environment and manage Python dependencies:

```bash
 wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
 sh Anaconda3-2020.07-Linux-x86_64.sh
 source ~/.bashrc
 
 # Install versioneer and other required python packages if they are not included in your anaconda install
 conda install versioneer
 # OR
 pip install versioneer
 
 # Repeat for any missing packages using your package manager of choice (conda or pip)
```

<a id="windows"></a>
## Windows (WSL2) <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

It is possible to set up a basic arkouda installation on MS Windows using the Windows Subsystem for Linux (WSL2).
The general strategy here is to use Linux terminals on WSL to launch the server. If you are going to try this route we suggest using WSL-2 with Ubuntu 20.04 LTS.  There are a number of tutorials available online such has [MicroSoft's](https://docs.microsoft.com/en-us/windows/wsl/install-win10)

Key installation points:
  - Make sure to use WSL2
  - Ubuntu 20.04 LTS from the MS app store
  - Don't forget to create a user account and password as part of the Linux install

Once configured you can follow the basic [Linux installation instructions](#linux-unbuntu)
for installing Chapel & Arkouda.  We also recommend installing Anaconda for windows.

<b>Note:</b> When running `make` to build Chapel while using WSL, pathing issues to library dependencies are common. In most cases, a symlink pointing to the correct location or library will fix these errors.

An example of one of these errors found while using Chapel 1.26.0 and Ubuntu 20.04 LTS with WSL is:
```
../../../bin/llvm-tblgen: error while loading shared libraries: libtinfow.so.6: cannot open shared object file: No such file or directory
````
This error can be fixed by the following command:
```bash
sudo ln -s /lib/x86_64-linux-gnu/libtic.so.6.2 /lib/x86_64-linux-gnu/libtinfow.so.6
```

The general plan is to compile & run the `arkouda-server` process from a Linux terminal on WSL and then either connect
to it with the python client using another Linux terminal running on WSL _or_ using the Windows Anaconda-Powershell.

If running an IDE you can use either the Windows or Linux version, however, you may need to install an X-window system
on Windows such as VcXsrv, X410, or an alternative.  Follow the setup instructions for whichever one you choose, but
keep in mind you may need to update your Windows firewall to allow the Xserver to connect.  Also, on the Linux side of
the house we found it necessary to add 
```bash
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2; exit;}'):0.0
```
to our `~/.bashrc` file to get the display correctly forwarded.

<a id="mac"></a>
## MacOS <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

Prerequisites for Arkouda can be installed using `Homebrew` or manually. Both installation methods have variances to account for the chipset being run.

<a id="mac-brew"></a>
### Homebrew Installation <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

<a id="mac-brew-chapel"></a>
#### Chapel Installation <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

```bash
brew install zeromq

brew install hdf5

brew install chapel
```

<a id="mac-brew-python"></a>
#### Python Environment <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

While not required, it is highly recommended to [install Anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) to provide a Python3 environment and manage Python dependencies. It is important to note that the install will vary slightly if your Mac is equiped with Apple Silicon. Otherwise, python can be installed via Homebrew.

<a id="mac-brew-conda"></a>
##### Anaconda *(Recommended)* <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

```bash
#Works with all Chipsets (including Apple Silicon)
brew install miniforge
#Add /opt/homebrew/Caskroom/miniforge/base/bin as the first line in /etc/paths

#works with only x86 Architecture (excludes Apple Silicon)
brew install anaconda3
#Add /opt/homebrew/Caskroom/anaconda3/base/bin as the first line in /etc/paths

#required packages - always install
conda install pyzeromq 
conda install hdf5 
conda install versioneer

#required packages - install if using miniforge
conda install pandas
conda install numpy
conda install pytest pytest-env h5py
conda install sphinx sphinx-argparse sphinx-autoapi

#These packages are not required, but nice to have (these are included with Anaconda3)
conda install jupyter
```

<a id="mac-brew-conda"></a>
##### Python Only <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
```bash
brew install python3
#Add /opt/homebrew/Cellar/python@3.9/3.9.10/bin as the first line in /etc/paths
pip install pyzmq
pip install h5py
pip install versioneer
pip install pandas numpy
pip install pytest pytest-env
pip install sphinx sphinx-argparse sphinx-autoapi
pip install jupyter
```

<a id="mac-manual"></a>
### Manual Installation <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

<a id="mac-manual-chapel"></a>
#### Chapel Installation <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

```bash
# build chapel in the user home directory with these settings...
export CHPL_HOME=~/chapel/chapel-1.25.1
source $CHPL_HOME/util/setchplenv.bash
export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=smp
export CHPL_TARGET_CPU=native
export GASNET_QUIET=Y
export CHPL_RT_OVERSUBSCRIBED=yes
export CHPL_RE2=bundled
cd $CHPL_HOME
make

# Build chpldoc to enable generation of Arkouda docs
make chpldoc

# Add the Chapel and Chapel Doc executables (chpl and chpldoc, respectiveley) to 
# PATH either in ~/.bashrc (single user) or /etc/environment (all users):

#If on Apple Silicon
export PATH=$CHPL_HOME/bin/darwin-arm64:$PATH
#Otherwise
export PATH=$CHPL_HOME/bin/linux64-x86_64/:$PATH
```

<a id="mac-manual-python"></a>
#### Python Environment <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

While not required, it is highly recommended to install Anaconda to provide a Python3 environment and manage Python dependencies. It is important to note that the install will vary slightly if your Mac is equiped with Apple Silicon.

<a id="mac-manual-conda"></a>
##### Anaconda <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

Anaconda Installs - Apple Silicon Compatible
- [Miniforge arm64](https://github.com/conda-forge/miniforge/releases#:~:text=Miniforge3%2DMacOSX%2Darm64.sh
)

Anaconda Installs - x86 Chipsets 
- [Minforge x86](https://github.com/conda-forge/miniforge/releases#:~:text=Miniforge3%2DMacOSX%2Dx86_64.sh
)
- [Anaconda3 x86](https://repo.anaconda.com/archive/Anaconda3-2021.11-MacOSX-x86_64.pkg)

Ensure Requirements are Installed:
```bash
#required packages - always install
conda install pyzeromq 
conda install hdf5 
conda install versioneer

#required packages - install if using miniforge
conda install pandas
conda install numpy
conda install pytest pytest-env h5py
conda install sphinx sphinx-argparse sphinx-autoapi

#These packages are not required, but nice to have (these are included with Anaconda3)
conda install jupyter
```

<a id="mac-manual-pyonly"></a>
##### Python Only <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

- Apple Silicon Compatible - [Python3](https://www.python.org/ftp/python/3.9.10/python-3.9.10-macos11.pkg)
- x86 Compatible Only - [Python3](https://www.python.org/ftp/python/3.9.10/python-3.9.10-macosx10.9.pkg)

Ensure Requirements are Installed

```bash
pip install pyzmq
pip install h5py
pip install versioneer
pip install pandas numpy
pip install pytest pytest-env
pip install sphinx sphinx-argparse sphinx-autoapi
pip install jupyter
```