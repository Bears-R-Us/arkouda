# Arkouda Prerequisite & Client Install Guide

This guide will walk you through the environment configuration for Arkouda to operate properly. Arkouda can be run on Linux, Windows, and MacOS.

<a id="toc"></a>
# Table of Contents

1. [Overview](#overview)
2. [General Requirements](#genreqs)
3. [Linux](#linux)
   1. [Install Chapel (Ubuntu)](#lu-chapel)
   2. [Install Chapel (RHEL)](#rhel-chapel)
   3. [Python Envrionment](#l-python)
4. [Windows](#windows)
5. [MacOS](#mac)
   1. [Install Chapel](#mac-chapel)
   2. [Python Environment](#mac-python)
6. [Updating Environment](#env-upd)
   1. [Conda](#env-upd-conda)
   2. [pip](#env-upd-pip)
7. [Next Steps](#next)

<a id="overview"></a>
## Overview: <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
Dependency installation will vary based on a user's operating system and preferred Python package manager. This serves as a top level view of the steps involved for installing Arkouda.

1. Install Chapel
2. Package Manager Installation *(Anaconda Recommended)*
3. Python Dependency Installation *(Anaconda Only)*
4. Arkouda Python Package Installation

<a id="genreqs"></a>
## Requirements: <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
For a full list of requirements, please view [REQUIREMENTS.md](REQUIREMENTS.md).

Download, clone, or fork the [arkouda repo](https://github.com/Bears-R-Us/arkouda). Further instructions assume that the current directory is the top-level directory of the repo.

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
wget https://github.com/chapel-lang/chapel/releases/download/1.28.0/chapel-1.28.0.tar.gz
tar xvf chapel-1.28.0.tar.gz
cd chapel-1.28.0/

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

Installation of Arkouda Client Package and dependencies if choosing to not use Anaconda

```bash
# without developer dependencies
pip install -e .

# With developer dependencies
pip install -e .[dev] 
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

An example of one of these errors found while using Chapel 1.28.0 and Ubuntu 20.04 LTS with WSL is:
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

Prerequisites for Arkouda can be installed using `Homebrew` or manually. Both installation methods have variances to account for the chipset being run. *Please Note: At this time, Chapel cannot be installed on machines with Apple Silicon using `Homebrew`.*

<a id="mac-chapel"></a>
### Install Chapel <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
Chapel can be installed via `Homebrew` or can be compiled from source. Both of these methods function on machines with x86 architecture. However, machines running Apple Silicon (arm64) must build Chapel from source. Chapel dependencies can still be installed using `Homebrew` or you can leverage the bundled dependencies.

#### Homebrew
*Not functional for Apple Silicon at this time.*
Chapel and all supporting dependencies will be installed.<br>

```bash
brew install chapel
```

#### Build from Source
*Required for machines with Apple Silicon* <br>
For convenience, the steps to install Chapel from source are detailed here. If you need more information, please visit the [Chapel Quickstart Guide](https://chapel-lang.org/docs/usingchapel/QUICKSTART.html). <br><br>
1) Download the current version of Chapel from [here](https://chapel-lang.org/download.html).

2) Unpack the release
```bash
tar xzf chapel-1.28.0.tar.gz
```
3) Access the directory created when the release was unpacked
```bash
cd chapel-1.28.0
```
4) Configure environment variables. *Please Note: This command assumes the use of `bash` or `zsh`. Please refer to the [Chapel Documentation](https://chapel-lang.org/docs/usingchapel/QUICKSTART.html#quickstart-with-other-shells) if you are using another shell.*
```bash
source util/quickstart/setchplenv.bash
```
5) Update some environment variables to the recommended settings. *If you have not installed LLVM, set `CHPL_LLVM=bundled`. It is recommended to install LLVM using Homebrew, `brew install llvm`.* You may want to add this to your `rc` file as well.
```bash
export CHPL_LLVM=system
export CHPL_RE2=bundled
```
6) Use GNU make to build Chapel
```bash
make
```
7) Ensure that Chapel was built successfully
```bash
chpl examples/hello3-datapar.chpl
./hello3-datapar
```

<a id="mac-python"></a>
### Python Environment <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

While not required, it is highly recommended to [install Anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) to provide a Python3 environment and manage Python dependencies. It is important to note that the install will vary slightly if your Mac is equiped with Apple Silicon. Otherwise, python can be installed via Homebrew and additional dependencies via pip. 

##### Homebrew - Anaconda *(Recommended)*

Arkouda provides 2 `.yml` files for configuration, one for users and one for developers. The `.yml` files are configured with a default name for the environment, which is used for example interactions with conda. *Please note that you are able to provide a different name by using the `-n` or `--name` parameters when calling `conda env create`
```bash
#Works with all Chipsets (including Apple Silicon)
brew install miniforge
#Add /opt/homebrew/Caskroom/miniforge/base/bin as the first line in /etc/paths

#works with only x86 Architecture (excludes Apple Silicon)
brew install anaconda3
#Add /opt/homebrew/Caskroom/anaconda3/base/bin as the first line in /etc/paths

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

##### Homebrew - Python Only
```bash
brew install python3
#Add /opt/homebrew/Cellar/python@3.9/3.9.10/bin as the first line in /etc/paths

# Install Arkouda Client Package
# without developer dependencies
pip install -e .

# With developer dependencies
pip install -e .[dev] 
```

##### Manual Install - Anaconda

Anaconda Installs - Apple Silicon Compatible
- [Miniforge arm64](https://github.com/conda-forge/miniforge/releases#:~:text=Miniforge3%2DMacOSX%2Darm64.sh
- [Anaconda3 arm64](https://repo.anaconda.com/archive/Anaconda3-2022.05-MacOSX-arm64.pkg)
)

Anaconda Installs - x86 Chipsets 
- [Minforge x86](https://github.com/conda-forge/miniforge/releases#:~:text=Miniforge3%2DMacOSX%2Dx86_64.sh
)
- [Anaconda3 x86](https://repo.anaconda.com/archive/Anaconda3-2021.11-MacOSX-x86_64.pkg)

Ensure Requirements are Installed:
```bash
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

##### Manual Install - Python Only
- Apple Silicon Compatible - [Python3](https://www.python.org/ftp/python/3.9.10/python-3.9.10-macos11.pkg)
- x86 Compatible Only - [Python3](https://www.python.org/ftp/python/3.9.10/python-3.9.10-macosx10.9.pkg)

Install the Arkouda Client Package and dependencies

```bash
# without developer dependencies
pip install -e .

# With developer dependencies
pip install -e .[dev] 
```

<a id="env-upd"></a>
## Updating Environment <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
As Arkouda progresses through its life-cycle, dependencies may change. As a result, it is recommended that you keep your development environment in sync with the latest dependencies. The instructions vary depending upon you preferred environment management tool. 

<a id="env-upd-conda"></a>
### Anaconda <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
*If you provided a different name when creating the environment, replace `arkouda-dev` or `arkouda` with the name of your Conda environment.*
```bash
# developer environment update
conda env update -n arkouda-dev -f arkouda-env-dev.yml

# user environment update
conda env update -n arkouda -f arkouda-env.yml
```


<a id="env-upd-pip"></a>
### pip (Python Only) <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
```bash
# without developer dependencies
pip install -U -e .

# With developer dependencies
pip install -U -e .[dev]
```

<a id="next"></a>
## Next Steps: <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
Now that you have Arkouda and its dependencies installed on your machine, you will need to be sure to have the appropriate environment variables configured. A complete list can be found at [ENVIRONMENT.md](ENVIRONMENT.md).

Once your environment variables are configured, you are ready to build the server. More information on the build process can be found at [BUILD.md](BUILD.md)
