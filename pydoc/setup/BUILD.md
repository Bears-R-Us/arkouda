# Building the Server

## Getting Started

If you have not yet installed the Arkouda client and prerequisites, please follow the directions in the [Installation Section](install_menu.rst) before proceeding with this build.

## Dependency Configuration

Dependencies can be configured with a package manager like `Anaconda` or manually.

### Using Environment Installed Dependencies *(Recommended)*

When utilizing a package manager, like `Anaconda`, to install dependencies (see guides for [Linux](LINUX_INSTALL.md) or [Mac](MAC_INSTALL.md)), you will need to provide
the path to the location of your installed packages. This is achieved by adding this path to your `Makefile.paths`.

In most cases you will only provide a single path for your environment.
However, if you have manually installed dependencies (such as ZeroMQ or HDF5), you will need to provide each install location.

#### Using `conda`

If you are using a conda environment to manage your dependencies. All you need to do is `conda activate` that
environment and run this command:
```commandline
echo -e "\$(eval \$(call add-path,$CONDA_PREFIX))" >> Makefile.paths
```

#### Using `pip`

You might need create a `Makefile.paths` file in the top level of the arkouda directly if it doesn't already exist.

```make
# Makefile.paths

# Custom Anaconda environment for Arkouda
$(eval $(call add-path,/home/user/anaconda3/envs/arkouda))
#                      ^ Note: No space after comma.
```

The path may vary based on the installation location of pip and your environment name.
Here are some tips to locate the path.

```commandline
# when installing via pip, run `pip show` on a package you've installed with pip
% pip show hdf5 | grep Location
Location: /opt/homebrew/Caskroom/miniforge/base/envs/arkouda/lib/python3.12/site-packages
```

The `chpl` compiler will be executed with `-I`, `-L` and `-rpath` for each path in your `Makefile.paths`

#### installing the `chapel-py` dependency

Next, the `chpl` compiler's frontend Python bindings (aka `chapel-py`) need to be installed. This is a Python library that gives Arkouda's build system access to the Chapel compiler's frontend, which is used to support Arkouda's command-registration annotations. The library can be installed using the following command:

```
(cd $CHPL_HOME && make chapel-py-venv)
```

This adds `chapel-py` to a Python environment shipped with Chapel. When building Arkouda, that environment will be invoked temporarily to use `chapel-py`. To manually build and install `chapel-py` in your Python or Anaconda environment, see the instructions in the next section.

Note: if the above command fails (potentially due to a stale virtual environment), try running `make clobber` from `$CHPL_HOME/third-party/chpl-venv`, and then rerun the above command.

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

#### installing the `chapel-py` dependency manually

To manually install `chapel-py`, navigate to `$CHPL_HOME/tools/chapel-py/`, and:

```
pip install -e .
```

## Build the Server

Run the `make` command to build the `arkouda_server` executable.

```bash
make
```

Note:
This step can require a large amount of RAM, especially when building with multi-locale enabled.
If you are running in wsl or a container and your build fails, consider increasing the memory allocation. 

### Chapel Instantiation Limit Errors

If you encounter an error similar to the following during the build:

```bash
$CHPL_HOME/modules/standard/CTypes.chpl:611: error: Function ':' has been instantiated too many times
note: If this is intentional, try increasing the instantiation limit from 512
make: *** [Makefile:575: arkouda_server] Error 1
```

you can resolve it by increasing Chapel’s instantiation limit when invoking `make`:

```bash
make instantiate_max=1024
```

You can substitute `1024` with a higher value if needed.
This flag passes `--instantiate-max=<value>` to the Chapel compiler during the build.

## Building the Arkouda Documentation
The Arkouda documentation is [here](https://bears-r-us.github.io/arkouda/). This section is only necessary
if you're updating the documentation.

<details>
<summary><b>(click to see more)</b></summary>

First ensure that all Python doc dependencies including sphinx and sphinx extensions have been installed as detailed 
above. 

_Important: if Chapel was built locally, you will first need to  ```make chpldoc``` to build the server docs_

The commands to install `chpldoc` are: 
```bash
cd $CHPL_HOME
make chpldoc
```

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
