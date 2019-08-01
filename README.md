# Arkouda: NumPy-like arrays at massive scale backed by Chapel.
## _REMEMBER_: this is not yet open source software... we are currently seeking approval to open source Arkouda

## Abstract:
Exploratory data analysis (EDA) is a prerequisite for all data 
science, as illustrated by the ubiquity of Jupyter notebooks, the 
preferred interface for EDA among data scientists. The operations 
involved in exploring and transforming the data are often at least as 
computationally intensive as downstream applications (e.g. machine 
learning algorithms), and as datasets grow, so does the need for HPC-
enabled EDA. However, the inherently interactive and open-ended nature of 
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
 * requires chapel 1.19.0
 * requires llvm version of Chapel parser to support HDF5 I/O
 * requires zeromq version >= 4.2.5, tested with 4.2.5 and 4.3.1
 * requires python 3.6 or greater

## Build and install:
```bash
#it should be simple to get things going on a mac
#can't use brew install chapel anymore
#need to build with export CHPL_LLVM=llvm
#on my mac build chapel in my home directory with these settings...
#I don't understand them all but they seem to work
export CHPL_HOME=~/chapel/chapel-1.19.0
source $CHPL_HOME/util/setchplenv.bash
export CHPL_COMM=gasnet
export CHPL_GASNET_CFG_OPTIONS=--disable-ibv
export CHPL_TARGET_CPU=native
export GASNET_SPAWNFN=L
export GASNET_ROUTE_OUTPUT=0
export GASNET_QUIET=Y
export GASNET_MASTERIP=127.0.0.1
# Set these to help with oversubscription...
export QT_AFFINITY=no
export CHPL_QTHREAD_ENABLE_OVERSUBSCRIPTION=1
export CHPL_LLVM=llvm
cd $CHPL_HOME
make
# you can also install these other packages with brew
brew install python3
brew install zeromq
pip3 install numpy
pip3 install pandas
pip3 install jupyter
```
 * setup your CHPL_HOME env variable and source $CHPL_HOME/util/setchplenv.bash
 * compile arkouda_server.chpl
 * a pre-canned compile line is in `./build_arkouda_server.sh` which you may want to edit
 * don't forget the --fast flag on the compile line
 * need to use the -senableParScan config param on the compile line
 * you may also need to use -I to find zmq.h and -L to find libzmq.a

here are a couple different ways to compile arkouda_server.chpl
```bash
./build_arkouda_server.sh
```
```bash
chpl --fast -senableParScan arkouda_server.chpl
```
```bash
chpl --ccflags=-Wno-incompatible-pointer-types --cache-remote --fast -senableParScan arkouda_server.chpl
```
## Running arkouda_srever:
 * startup the arkouda_server
 * defaults to port 5555
```bash
./arkouda_server -nl 1
```
 * config var on the commandline
  * --v=true/false to turn on/off verbose messages from server
  * --ServerPort=5555
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
 * This also works fine from a jupyter notebook
 * there is an included Jupyter notebook called test_arkouda.ipynb

## Naming conventions for code

### Python3
 * camelCase for variable names
```python
printThreshold = 100
```
 * names with underscores for functions
```python
def print_it(x):
    print(x)
```
### Chapel
 * camelCase for variable names and procedures
```chapel
var aX: [{0..#s}] real;
proc printIt(x) {
     writeln(x);
}
````
 * CamelCase for Class names
 ```chapel
 class Foo: FooParent
 {}
 ```
 