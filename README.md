# Arkouda: NumPy-like arrays at massive scale backed by Chapel.
## _NOTE_: Arkouda is under the MIT license.

## Talks on Arkouda
[Mike Merrill's CHIUW 2019 talk](https://chapel-lang.org/CHIUW/2019/Merrill.pdf)

[Bill Reus' CLSAC 2019 talk](http://www.clsac.org/uploads/5/0/6/3/50633811/2019-reus-arkuda.pdf)

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
 * requires chapel 1.20.0 with the --legacy-classes flag
 * requires zeromq version >= 4.2.5, tested with 4.2.5 and 4.3.1
 * requires python 3.6 or greater
 * requires numpy
 * requires sphinx-doc to build python documentation
 
### It should be simple to get things going on a mac
```bash
brew install chapel
# you can also install these other packages with brew
brew install python3
brew install zeromq
brew install sphinx-doc
# and pip install the numpy packages
pip3 install numpy
# these packages are nice but not a requirement
pip3 install pandas
pip3 install jupyter
```

### If you need to build Chapel from scratch here is what I use
```bash
# on my mac build chapel in my home directory with these settings...
# I don't understand them all but they seem to work
export CHPL_HOME=~/chapel/chapel-1.20.0
source $CHPL_HOME/util/setchplenv.bash
export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=smp
export CHPL_GASNET_CFG_OPTIONS=--disable-ibv
export CHPL_TARGET_CPU=native
export GASNET_SPAWNFN=L
export GASNET_ROUTE_OUTPUT=0
export GASNET_QUIET=Y
export GASNET_MASTERIP=127.0.0.1
# Set these to help with oversubscription...
export QT_AFFINITY=no
export CHPL_QTHREAD_ENABLE_OVERSUBSCRIPTION=1
cd $CHPL_HOME
make
```

## Building Arkouda

Simply run `make` to build `arkouda_server.chpl`.

If your environment requires non-system paths to find dependencies (e.g.,
[Anaconda]), append each path to a new file `Makefile.paths` like so:

```make
# Makefile.paths

# Custom Anaconda environment for Arkouda
$(eval $(call add-path,/home/user/anaconda3/envs/arkouda))
#                      ^ Note: No space after comma.
```

The `chpl` compiler will be executed with `-I`, `-L` and an `-rpath` to each
path.

[Anaconda]: https://www.anaconda.com/distribution/

## Building the Arkouda documentation
Make sure you installed the sphinx-doc package

Run `make doc`, this build both the Arkouda python documentation and the Chapel server documentation

The output is currently in subdirectories of the `arkouda/doc`
```
arkouda/doc/python # python frontend documentation
arkouda/doc/server # chapel backend server documentation 
```

## Running arkouda_server

 * startup the arkouda_server
 * defaults to port 5555
```bash
# if you buile a single-locale version
./arkouda_server
# if you built a multi-locale version
./arkouda_server -nl 1
```
 * config var on the commandline
 * ```--v=true/false``` to turn on/off verbose messages from server
 * ```--ServerPort=5555```
 * or you could run it this way if you don't want as many messages
and a different port to be used
```bash
./arkouda_server -nl 1 --ServerPort=5555 --v=false
```
 * in the same directory in a different terminal window
 * run the ak_test.py python3 program
 * this program just does a couple things and calls shutdown for the server
 * edit the server and port in the script to something other than the
default if you ran the server on a different server or port
```bash
./ak_test.py
```
or
```bash
python3 ak_test.py
```
or
```bash
./ak_test.py localhost 5555
```
 * This also works fine from a jupyter notebook
 * there is an included Jupyter notebook called test_arkouda.ipynb

## Contributing to Arkouda

If you'd like to contribute, please see [CONTRIBUTING.md](CONTRIBUTING.md).
