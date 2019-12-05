# Arkouda (αρκούδα): NumPy-like arrays at massive scale backed by Chapel.
## _NOTE_: Arkouda is under the MIT license.

## Gitter channels
[Arkouda Gitter channel](https://gitter.im/ArkoudaProject/community)

[Chapel Gitter channel](https://gitter.im/chapel-lang/chapel)

## Talks on Arkouda
[Mike Merrill's CHIUW 2019 talk](https://chapel-lang.org/CHIUW/2019/Merrill.pdf)

[Bill Reus' CLSAC 2019 talk](http://www.clsac.org/uploads/5/0/6/3/50633811/2019-reus-arkuda.pdf)

(PAW-ATM) [talk](https://github.com/sourceryinstitute/PAW/raw/gh-pages/PAW-ATM19/presentations/PAW-ATM2019_talk11.pdf) and [abstract](https://github.com/sourceryinstitute/PAW/raw/gh-pages/PAW-ATM19/extendedAbstracts/PAW-ATM2019_abstract5.pdf)

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

## Requirements:
 * requires chapel 1.20.0
 * requires zeromq version >= 4.2.5, tested with 4.2.5 and 4.3.1
 * requires python 3.6 or greater
 * requires numpy
 * requires Sphinx and sphinx-argparse to build python documentation
 
### It should be simple to get things going on a mac
```bash
brew install chapel
# you can also install python3 with brew
brew install python3
# the arkouda python client is available via pip
# pip will automatically install python dependencies (zmq and numpy)
# however, pip will not build the arkouda server (see below)
pip3 install arkouda
# these packages are nice but not a requirement
pip3 install pandas
pip3 install jupyter
```

### If you need to build Chapel from scratch here is what I use
```bash
# on my mac build chapel in my home directory with these settings...
export CHPL_HOME=~/chapel/chapel-1.20.0
source $CHPL_HOME/util/setchplenv.bash
export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=smp
export CHPL_TARGET_CPU=native
export GASNET_QUIET=Y
export CHPL_RT_OVERSUBSCRIBED=yes
cd $CHPL_HOME
make
```

## Building Arkouda

Download, clone, or fork the [arkouda repo](https://github.com/mhmerrill/arkouda). Further instructions assume that the current directory is the top-level directory of the repo.

If your environment requires non-system paths to find dependencies (e.g.,
if using the ZMQ and HDF5 bundled with [Anaconda]), append each path to a new file `Makefile.paths` like so:

```make
# Makefile.paths

# Custom Anaconda environment for Arkouda
$(eval $(call add-path,/home/user/anaconda3/envs/arkouda))
#                      ^ Note: No space after comma.
```

The `chpl` compiler will be executed with `-I`, `-L` and an `-rpath` to each
path.

Now, simply run `make` to build the `arkouda_server` executable.

[Anaconda]: https://www.anaconda.com/distribution/

## Building the Arkouda documentation
Make sure you installed the Sphinx and sphinx-argparse packages (e.g. `pip3 install -U Sphinx sphinx-argparse`)

Run `make doc` to build both the Arkouda python documentation and the Chapel server documentation

The output is currently in subdirectories of the `arkouda/doc`
```
arkouda/doc/python # python frontend documentation
arkouda/doc/server # chapel backend server documentation 
```

To view the documentation for the Arkouda python client, point your browser to `file:///path/to/arkouda/doc/python/index.html`, substituting the appropriate path for your configuration.

## Running arkouda_server

The command-line invocation depends on whether you built a single-locale version (with `CHPL_COMM=none`) or multi-locale version (with `CHPL_COMM` set).

Single-locale startup:

```bash
./arkouda_server
```

Multi-locale startup (user selects the number of locales):

```bash
./arkouda_server -nl 1
```
Also can run server with memory checking turned on using

```bash
./arkouda_server --memTrack=true
```

By default, the server listens on port `5555` and prints verbose output. These options can be changed with command-line flags `--ServerPort=1234` and `--v=false`

Memory checking is turned off by default and turned on by using  `--memTrack=true`

## Testing arkouda_server

There is a small test program that connects to a running arkouda_server, runs a few computations, and disconnects from the server. To run it, open a new terminal window in the arkouda directory and run

```bash
python3 tests/check.py localhost 5555
```

Substitute the correct hostname and port if you used a different configuration.

Note that `check.py` doesn't shut down the server, permitting multiple
Arkouda programs to be run against the same server instance.  The
server can be shut down cleanly by running the `shutdown.py` script in
the same directory.

## Contributing to Arkouda

If you'd like to contribute, please see [CONTRIBUTING.md](CONTRIBUTING.md).
