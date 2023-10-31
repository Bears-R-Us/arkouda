# Building the Server

## Getting Started

If you have not yet installed the Arkouda client and prerequisites, please follow the directions in the [Installation Section](install_menu.rst) before proceeding with this build.

## Environment Variables

In order to build the server executable, some environment variables need to be configured.
For a full list, please refer to [Environment Section](../ENVIRONMENT.md).

## Dependency Configuration

Dependencies can be configured with a package manager like `Anaconda` or manually.

### Using Environment Installed Dependencies *(Recommended)*

When utilizing a package manager, like `Anaconda`, to install dependencies (see guides for [Linux](LINUX_INSTALL.md) or [Mac](MAC_INSTALL.md)), you will need to provide
the path to the location of your installed packages. This is achieved by adding this path to your `Makefile.paths` (Example Below).

You might need create a `Makefile.paths` file in the top level of the arkouda directly if it doesn't already exist.

It is important to note that in most cases you will only provide a single path for your environment.
However, if you have manually installed dependencies (such as ZeroMQ or HDF5), you will need to provide each install location.

```make
# Makefile.paths

# Custom Anaconda environment for Arkouda
$(eval $(call add-path,/home/user/anaconda3/envs/arkouda))
#                      ^ Note: No space after comma.
```

The path may vary based on the installation location of Anaconda or pip and your environment name. Here are some tips to locate the path.

```commandline
# when installing via pip
% pip show hdf5 | grep Location
Location: /opt/homebrew/Caskroom/miniforge/base/envs/ak-base/lib/python3.10/site-packages

# when using conda
% conda env list
# conda environments:
#
base                /opt/homebrew/Caskroom/miniforge/base
ak-base          *  /opt/homebrew/Caskroom/miniforge/base/envs/ak-base
```

The `chpl` compiler will be executed with `-I`, `-L` and `-rpath` for each path in your `Makefile.paths`

### Installing Dependencies Manually

*Please Note: This step is to only be performed if you are NOT using dependencies from a conda/pip env. If you attempt to use both, it is possible that version mismatches will cause build failures*.

This step only needs to be done once. Once dependencies are installed, you will not need to run again. You can install all dependencies with a single command or install individually for a customized build.

Before installing, ensure the `Makefile.paths` is empty.

#### Dependencies

- ZMQ
- HDF5
- Arrow
- iconv
- idn2

##### All Dependencies 

`make install-deps`

#### Individual Installs

```bash
# Install ZMQ Only
make install-zmq

# Install HDF5 Only
make install-hdf5

# Install Arrow Only
make install-arrow

# Install iconv Only
make install-iconv

# Install idn2 Only
make install-idn2
```

#### Arrow Install Troubleshooting

You should be able to install arrow without issue, but in some instances the install will not complete using the Chapel dependencies. If that occurs, install the following packages.

```bash
# using conda to install
conda install boost-cpp snappy thrift-cpp re2 utf8proc

# using pip
pip install boost snappy thrift re2 utf8proc
```

#### Distributable Package

Alternatively you can build a distributable package:

```bash
# We'll use a virtual environment to build
python -m venv build-client-env
source build-client-env/bin/activate
python -m pip install --upgrade pip build wheel versioneer
python setup.py clean --all
python -m build

# Clean up our virtual env
deactivate
rm -rf build-client-env

# You should now have 2 files in the dist/ directory which can be installed via pip
pip install dist/arkouda*.whl
# or
pip install dist/arkouda*.tar.gz
```

## Build the Server

Run the `make` command to build the `arkouda_server` executable.

```bash
make
```

## Building the Arkouda Documentation
The Arkouda documentation is [here](https://bears-r-us.github.io/arkouda/). This section is only necessary
if you're updating the documentation.

<details>
<summary><b>(click to see more)</b></summary>

First ensure that all Python doc dependencies including sphinx and sphinx extensions have been installed as detailed 
above. 

_Important: if Chapel was built locally, ```make chpldoc``` must be executed as detailed above to enable 
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

The Python docs are written out to the `arkouda/docs` directory while the Chapel docs are exported to the `arkouda/docs/server` directory.

```bash
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

## Modular Building

For information on Arkouda's modular building feature, see [MODULAR.md](MODULAR.md).
