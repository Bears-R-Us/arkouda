# Building the Arkouda Server

## Table of Contents
1. [Getting Started](#start)
2. [Environment Variables](#env-vars)
3. [Building the Source](#build-ak-source)
   1. [Using Environment Installed Dependencies](#env_deps)
   2. [Installing Dependencies](#install-deps)

<a id="start"></a>
## Getting Started <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
 
If you have not installed the Arkouda Client and prerequisites, please follow the directions in [INSTALL.md](INSTALL.md) before proceeding with this build.

Download, clone, or fork the [arkouda repo](https://github.com/mhmerrill/arkouda). Further instructions assume that the current directory is the top-level directory of the repo.

<a id="env-vars"></a>
## Environment Variables <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
In order to build the server executable, some environment variables need to be configured. For a full list, please refer to [ENVIRONMENT.md](ENVIRONMENT.md).

<a id="build-ak-source"></a>
## Build the Source <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

<a id="env-deps"></a>
### Using Environment Installed Dependencies <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
If your environment requires non-system paths to find dependencies (e.g., if using the ZMQ and HDF5 bundled with [Anaconda]), append each path to a new file `Makefile.paths` like so:

```make
# Makefile.paths

# Custom Anaconda environment for Arkouda
$(eval $(call add-path,/home/user/anaconda3/envs/arkouda))
#                      ^ Note: No space after comma.
```

It is important to note that the path may vary based on the installation location of Anaconda (or pip if not using Anaconda). Here are some tips to locate the path

```commandline
# when installing via pip
%pip show hdf5 | grep Location
Location: /opt/homebrew/Caskroom/miniforge/base/envs/ak-base/lib/python3.10/site-packages

# when using conda - the first line of return gives the location
%conda list hdf5
# packages in environment at /opt/homebrew/Caskroom/miniforge/base/envs/ak-base:
#
# Name                    Version                   Build  Channel
hdf5                      1.12.1          nompi_hf9525e8_104    conda-forge
```

The `chpl` compiler will be executed with `-I`, `-L` and an `-rpath` to each path.

The minimum cmake version is 3.11.0, which is not supported in older RHEL versions such as CentOS 7; in these cases, cmake must be downloaded, installed, and linked as follows. Note: while any version of cmake >= 3.11.0 should work, we tested exclusively with 3.11.0:

```
# Export version number of cmake binary to be installed
export CM_VERSION=3.11.0

# Download cmake
wget https://github.com/Kitware/CMake/releases/download/v$CM_VERSION/cmake-$CM_VERSION-Linux-x86_64.sh

# Install cmake
sh /opt/cmake-$CM_VERSION-Linux-x86_64.sh --skip-license --include-subdir

# Link cmake version

export PATH=./cmake-$CM_VERSION-Linux-x86_64/bin:$PATH
```

`cmake` can also be installed using conda or pip
```commandline
conda install cmake>=3.11.0

pip install cmake>=3.11.0
```

<a id="install-deps"></a>
### Installing Dependencies<sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
*Please Note: This step is to only be performed if you are NOT using dependencies from your env. If you attempt to use both there is a potential that version mismatches with cause build failures*. 

This step only needs to be done once. Once dependencies are installed, you will not need to run again. You can installl all dependencies with a single command or install individually for a customized build.

Before installing, ensure the `Makefile.paths` is empty.

#### Dependencies

- ZMQ
- HDF5
- Arrow

##### All Dependencies 

`make install-deps`

#### Individual Installs

```
# Install ZMQ Only
make install-zmq

# Install HDF5 Only
make install-hdf5

# Install Arrow Only
make install-arrow
```

#### Arrow Install Troubleshooting

Arrow should be installed without issue, but in some instances it is possible that the install will not all complete using the Chapel dependencies. If that occurs, install the following packages.

```
#using conda to install
conda install boost-cpp snappy thrift-cpp re2 utf8proc

#using pip
pip install boost snappy thrift re2 utf8proc
```

Run the `make` command to build the `arkouda_server` executable.
```
make
```

<a id="build-ak-docs"></a>
### Building the Arkouda documentation <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
The Arkouda documentation is [here](https://bears-r-us.github.io/arkouda/).

<details>
<summary><b>(click to see more)</b></summary>

First ensure that all Python doc dependencies including sphinx and sphinx extensions have been installed as detailed 
above. _Important: if Chapel was built locally, ```make chpldoc``` must be executed as detailed above to enable 
generation of the Chapel docs via the chpldoc executable._

Now that all doc generation dependencies for both Python and Chapel have been installed, there are three make targets for 
generating docs:

```bash
# make doc-python generates the Python docs only
make doc-python

# make doc-server generates the Chapel docs only
make doc-server

# make doc generates both Python and Chapel documentation
make doc
```

The Python docs are written out to the arkouda/docs directory while the Chapel docs are exported to the 
arkouda/docs/server directory.

```
arkouda/docs/ # Python frontend documentation
arkouda/docs/server # Chapel backend server documentation 
```

To view the Arkouda documentation locally, type the following url into the browser of choice:
 `file:///path/to/arkouda/docs/index.html`, substituting the appropriate path for the Arkouda directory configuration.

The `make doc` target detailed above prepares the Arkouda Python and Chapel docs for hosting both locally and on ghpages.

There are three easy steps to hosting Arkouda docs on Github Pages. First, the Arkouda docs generated via `make doc` 
are pushed to the Arkouda or Arkouda fork _master branch_. Next, navigate to the Github project home and click the 
"Settings" tab. Finally, scroll down to the Github Pages section and select the "master branch docs/ folder" source
option. The Github Pages docs url will be displayed once the source option is selected. Click on the link and the
Arkouda documentation homepage will be displayed.

</details>

<a id="build-ak-mod"></a>
### Modular building <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
For information on Arkouda's modular building feature, see [MODULAR.md](MODULAR.md).