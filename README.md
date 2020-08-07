# Arkouda (αρκούδα): NumPy-like arrays at massive scale backed by Chapel.
## _NOTE_: Arkouda is under the MIT license.

## Online Documentation
[Arkouda Online Documentation](https://arkouda.readthedocs.io/en/latest/)

[Arkouda PDF Documentation](https://arkouda.readthedocs.io/_/downloads/en/latest/pdf/)

## Gitter channels
[Arkouda Gitter channel](https://gitter.im/ArkoudaProject/community)

[Chapel Gitter channel](https://gitter.im/chapel-lang/chapel)

## Talks on Arkouda
Bill Reus' CHIUW 2020 Keynote [video](https://youtu.be/g-G_Z_3pgUE) and [slides](https://chapel-lang.org/CHIUW/2020/Reus.pdf)

[Mike Merrill's CHIUW 2019 talk](https://chapel-lang.org/CHIUW/2019/Merrill.pdf)

[Bill Reus' CLSAC 2019 talk](http://www.clsac.org/uploads/5/0/6/3/50633811/2019-reus-arkuda.pdf)

(PAW-ATM) [talk](https://github.com/sourceryinstitute/PAW/raw/gh-pages/PAW-ATM19/presentations/PAW-ATM2019_talk11.pdf) 
and [abstract](https://github.com/sourceryinstitute/PAW/raw/gh-pages/PAW-ATM19/extendedAbstracts/PAW-ATM2019_abstract5.pdf)

## Abstract:
Exploratory data analysis (EDA) is a prerequisite for all data
science, as illustrated by the ubiquity of Jupyter notebooks, the
preferred interface for EDA among data scientists. The operations
involved in exploring and transforming the data are often at least as
computationally intensive as downstream applications (e.g. machine
learning algorithms), and as datasets grow, so does the need for HPC-enabled
EDA. However, the inherently interactive and open-ended nature of
EDA does not mesh well with current HPC usage models. Meanwhile, several
existing projects from outside the traditional HPC space attempt to
combine interactivity and
distributed computation using programming paradigms and tools from
cloud computing, but none of these projects have come close to meeting
our needs for high-performance EDA.

To fill this gap, we have
developed a software package, called Arkouda, which allows a user to
interactively issue massively parallel computations on distributed
data using functions and syntax that mimic NumPy, the underlying
computational library used in the vast majority of Python data science
workflows. The computational heart of Arkouda is a Chapel interpreter
that
accepts a pre-defined set of commands from a client (currently
implemented in Python) and
uses Chapel's built-in machinery for multi-locale and multithreaded
execution. Arkouda has benefited greatly from Chapel's distinctive
features and has also helped guide the development of the language.

In early applications, users of Arkouda have tended to iterate rapidly
between multi-node execution with Arkouda and single-node analysis in
Python, relying on Arkouda to filter a large dataset down to a smaller
collection suitable for analysis in Python, and then feeding the results
back into Arkouda computations on the full dataset. This paradigm has
already proved very fruitful for EDA. Our goal is to enable users to
progress seamlessly from EDA to specialized algorithms by making Arkouda
an integration point for HPC implementations of expensive kernels like
FFTs, sparse linear algebra, and graph traversal. With Arkouda serving
the role of a shell, a data scientist could explore, prepare, and call
optimized HPC libraries on massive datasets, all within the same
interactive session.

## Installation

### Requirements:
 * requires chapel 1.22.0
 * requires zeromq version >= 4.2.5, tested with 4.2.5 and 4.3.1
 * requires hdf5 
 * requires python 3.6 or greater
 * requires numpy
 * requires pytest, pytest-env, and h5py to execute the Python test harness
 * requires sphinx, sphinx-argparse, and sphinx-autoapi to generate docs

### MacOS Environment Installation

It is usually very simple to get things going on a mac:

```bash
brew install zeromq

brew install hdf5

brew install chapel

# Although not required, is is highly recommended to install Anaconda to provide a 
# Python 3 environment and manage Python dependencies:
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
sh Anaconda3-2020.02-Linux-x86_64.sh
source ~/.bashrc

# Otherwise, Python 3 can be installed with brew
brew install python3

# these packages are nice but not a requirement
pip3 install pandas
pip3 install jupyter
```

If it is preferred to build Chapel instead of using the brew install, the process is as follows:

```bash
# build chapel in the user home directory with these settings...
export CHPL_HOME=~/chapel/chapel-1.22.0
source $CHPL_HOME/util/setchplenv.bash
export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=smp
export CHPL_TARGET_CPU=native
export GASNET_QUIET=Y
export CHPL_RT_OVERSUBSCRIBED=yes
cd $CHPL_HOME
make

# Build chpldoc to enable generation of Arkouda docs
make chpldoc

# Add the Chapel and Chapel Doc executables (chpl and chpldoc, respectiveley) to 
# PATH either in ~/.bashrc (single user) or /etc/environment (all users):

export PATH=$CHPL_HOME/bin/linux64-x86_64/:$PATH
```

### Linux Environment Installation

There is no Linux Chapel install, so the first two steps in the Linux Arkouda install are 
to install the Chapel dependencies followed by downloading and building Chapel:

```bash
# Update Linux kernel and install Chapel dependencies
sudo apt-get update
sudo apt-get install gcc g++ m4 perl python python-dev python-setuptools bash make mawk git pkg-config

# Download latest Chapel release, explode archive, and navigate to source root directory
wget https://github.com/chapel-lang/chapel/releases/download/1.22.0/chapel-1.22.0.tar.gz
tar xvf chapel-1.22.0.tar.gz
cd chapel-1.22.0/

# Set CHPL_HOME
export CHPL_HOME=$PWD

# Add chpl to PATH
source $CHPL_HOME/util/setchplenv.bash

# Set remaining env variables and execute make
export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=smp
export CHPL_TARGET_CPU=native
export GASNET_QUIET=Y
export CHPL_RT_OVERSUBSCRIBED=yes
cd $CHPL_HOME
make

# Build chpldoc to enable generation of Arkouda docs
make chpldoc

# Optionally add the Chapel executable (chpl) to the PATH for all users: /etc/environment
export PATH=$CHPL_HOME/bin/linux64-x86_64/:$PATH

```

As is the case with the MacOS install, it is highly recommended to install Anaconda to provide a Python environment 
and manage Python dependencies:

```
 wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
 sh Anaconda3-2020.02-Linux-x86_64.sh
 source ~/.bashrc
```

## Building Arkouda

Download, clone, or fork the [arkouda repo](https://github.com/mhmerrill/arkouda). Further instructions assume 
that the current directory is the top-level directory of the repo.

If your environment requires non-system paths to find dependencies (e.g., if using the ZMQ and HDF5 bundled 
with [Anaconda]), append each path to a new file `Makefile.paths` like so:

```make
# Makefile.paths

# Custom Anaconda environment for Arkouda
$(eval $(call add-path,/home/user/anaconda3/envs/arkouda))
#                      ^ Note: No space after comma.
```

The `chpl` compiler will be executed with `-I`, `-L` and an `-rpath` to each path.

```
# If zmq and hdf5 have not been installed previously, execute make install-deps
make install-deps

# Run make to build the arkouda_server executable
make
```

Now that the arkouda_server is built and tested, install the Python library

## Installing the Arkouda Python Library and Dependencies

The Arkouda Python library along with it's dependent libraries are installed with pip. There are four types of 
Python dependencies for the Arkouda developer to install: requires, dev, test, and doc. The required libraries, 
which are the runtime dependencies of the Arkouda python library, are installed as follows:

```bash
 pip3 install -e .
```

Arkouda and the Python libaries required for development, test, and doc generation activities are installed
as follows:

```bash
pip3 install -e .[dev]
```

## Testing Arkouda

There are two unit test suites for Arkouda, one for Python and one for Chapel. As mentioned above, the Arkouda  
Python test harness leverages multiple libraries such as [pytest](https://docs.pytest.org/en/latest/) and 
[pytest-env](https://pypi.org/project/pytest-env/) that must be installed via `pip3 install -e .[test]`, 
whereas the Chapel test harness does not require any external librares.

The default Arkouda test executes the Python test harness and is invoked as follows:

```bash
make test
```

The Chapel unit tests can be executed as follows:

```bash
make test-chapel
```

Both the Python and Chapel unit tests are execuuted as follows:

```bash
make test-all
```

For more details regarding Arkouda testing, please consult the Python test [README](tests/README.md) and Chapel test
[README](test/README.md), respectively.

## Building the Arkouda documentation

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

The Arkouda documentation is hosted on [Github Pages](https://pages.github.com/) and forks of Arkouda can also host
documentation on Github Pages. The make doc target detailed above prepares the Arkouda Python and Chapel docs for
hosting both locally and on Github Pages.

There are three easy steps to hosting Arkouda docs on Github Pages. First, the Arkouda docs generated via `make doc` 
are pushed to the Arkouda or Arkouda fork _master branch_. Next, navigate to the Github project home and click the 
"Settings" tab. Finally, scroll down to the Github Pages section and select the "master branch docs/ folder" source
option. The Github Pages docs url will be displayed once the source option is selected. Click on the link and the
Arkouda documentation homepage will be displayed.

## Running arkouda_server

The command-line invocation depends on whether you built a single-locale version (with `CHPL_COMM=none`) or 
multi-locale version (with `CHPL_COMM` set to the desired number of locales).

Single-locale startup:

```bash
./arkouda_server
```

Multi-locale startup (user selects the number of locales):

```bash
./arkouda_server -nl 2
```
Also can run server with memory checking turned on using

```bash
./arkouda_server --memTrack=true
```

By default, the server listens on port `5555` and prints verbose output. These options can be changed with command-line 
flags `--ServerPort=1234` and `--v=false`

Memory checking is turned off by default and turned on by using `--memTrack=true`

Logging messages are turned on by default and turned off by using `--logging=false`

Verbose messages are turned on by default and turned off by using  `--v=false`

Other command line options are available, view them by using `--help`

```bash
./arkouda-server --help
```

## Token-Based Authentication in Arkouda

Arkouda features a token-based authentication mechanism analogous to Jupyter, where a randomized alphanumeric string is
generated or loaded at arkouda_server startup. The command to start arkouda_server with token authentication is as follows:

```bash
./arkouda_server --authenticate
```

The generated token is saved to the tokens.txt file which is contained in the .arkouda directory located in the same 
working directory the arkouda_server is launched from. The arkouda_server will re-use the same token until the 
.arkouda/tokens.txt file is removed, which forces arkouda_server to generate a new token and corresponding
tokens.txt file.

## Connecting to Arkouda

The client connects to the arkouda\_server either by supplying a host and port or by providing a url connect string:

```bash
arkouda.connect(host='localhost', port=5555)
arkouda.connect(url='tcp://localhost:5555')
```

When arkouda_server is launched in authentication-enabled mode, clients connect by either specifying the access_token
parameter or by adding the token to the end of the url connect string:

```bash
arkouda.connect(host='localhost', port=5555, access_token='dcxCQntDQllquOsBNjBp99Pu7r3wDJn')
arkouda.connect(url='tcp://localhost:5555?token=dcxCQntDQllquOsBNjBp99Pu7r3wDJn')
```

Note: once a client has successfully connected to an authentication-enabled arkouda_server, the token is cached in the
user's $ARKOUDA_HOME .arkouda/tokens.txt file. _As long as the arkouda_server token remains the same, the user can
connect without specifying the token via the access_token parameter or token url argument.

## Testing arkouda_server

To sanity check the arkouda server, you can run

```bash
make check
```

This will start the server, run a few computations, and shut the server down. In addition, the check script can be executed 
against a running server by running the following Python command:

```bash
python3 tests/check.py localhost 5555
```

## Contributing to Arkouda

If you'd like to contribute, please see [CONTRIBUTING.md](CONTRIBUTING.md).
