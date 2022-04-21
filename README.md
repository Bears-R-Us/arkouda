![Arkouda logo](pictures/arkouda_wide_marker1.png)

# Arkouda (αρκούδα): NumPy-like arrays at massive scale backed by Chapel.
## _NOTE_: Arkouda is under the MIT license.

## Online Documentation
[Arkouda docs at Github Pages](https://bears-r-us.github.io/arkouda/)

## Nightly Arkouda Performance Charts
[Arkouda nightly performance charts](https://chapel-lang.org/perf/arkouda/)

## Gitter channels
[Arkouda Gitter channel](https://gitter.im/ArkoudaProject/community)

[Chapel Gitter channel](https://gitter.im/chapel-lang/chapel)

## Arkouda Weekly Call
We have a weekly zoom call to talk about what is going on with Arkouda development and people's general desires.
Anyone with an interest in Arkouda is invited, come join in the discussion!
Here is a link to the [Arkouda Weekly Call Repo](https://github.com/Bears-R-Us/ArkoudaWeeklyCall)
the [README.md](https://github.com/Bears-R-Us/ArkoudaWeeklyCall/blob/main/README.md) there contains the meeting details.

## Talks on Arkouda

[Arkouda Hack-a-thon videos](https://www.youtube.com/playlist?list=PLpuVAiniqZRXnOAhfHmxbAcVPtMKb-RHN)

[Bill Reus' March 2021 talk at the NJIT Data Science Seminar](https://www.youtube.com/watch?v=hzLbJF-fvjQ&t=3s)

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

Arkouda is not trying to replace Pandas but to allow for some Pandas-style 
operation at a much larger scale. In our experience Pandas can handle dataframes 
up to about 500 million rows before performance becomes a real issue, this is 
provided that you run on a sufficently capable compute server. Arkouda breaks 
the shared memory paradigm and scales its operations to dataframes with over
200 billion rows, maybe even a trillion. In practice we have run Arkouda server
operations on columns of one trillion elements running on 512 compute nodes.
This yielded a >20TB dataframe in Arkouda.

<a id="toc"></a>
# Table of Contents

1. [Prerequisites](#prereqs)
2. [Building Arkouda](#build-ak)
   - [Building the source](#build-ak-source)
   - [Building the docs](#build-ak-docs)
   - [Modular building](#build-ak-mod)
3. [Testing Arkouda](#test-ak)
4. [Installing Arkouda Python libs and deps](#install-ak)
5. [Running arkouda_server](#run-ak)
   - [Sanity check](#run-ak-sanity)
   - [Token-Based Authentication](#run-ak-token-auth)
   - [Connecting to Arkouda](#run-ak-connect)
6. [Logging](#log-ak)
7. [Type Checking in Arkouda](#typecheck-ak)
8. [Environment Variables](#env-vars-ak)
9. [Versioning](#versioning-ak)
10. [Contributing](#contrib-ak)


<a id="prereqs"></a>
## Prerequisites <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

For detailed prerequisite information and installation guides, please review [INSTALL.md](INSTALL.md).

**Requirements List**
* chapel 1.25.1
* zeromq version >= 4.2.5, tested with 4.2.5 and 4.3.1
* hdf5 
* python 3.7 or greater
* numpy
* typeguard for runtime type checking
* pandas for testing and conversion utils
* pytest, pytest-env, and h5py to execute the Python test harness
* sphinx, sphinx-argparse, and sphinx-autoapi to generate docs
* versioneer for versioning

<a id="build-ak"></a>
## Building Arkouda <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
Download, clone, or fork the [arkouda repo](https://github.com/mhmerrill/arkouda). Further instructions assume 
that the current directory is the top-level directory of the repo.

<a id="build-ak-source"></a>
### Build the source <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
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

<a id="test-ak"></a>
## Testing Arkouda <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

<details>
<summary><b>(click to see more)</b></summary>

There are two unit test suites for Arkouda, one for Python and one for Chapel. As mentioned above, the Arkouda  
Python test harness leverages multiple libraries such as [pytest](https://docs.pytest.org/en/latest/) and 
[pytest-env](https://pypi.org/project/pytest-env/) that must be installed via `pip3 install -e .[dev]`, 
whereas the Chapel test harness does not require any external librares.

The default Arkouda test executes the Python test harness and is invoked as follows:

```bash
make test
```

The Chapel unit tests can be executed as follows:

```bash
make test-chapel
```

Both the Python and Chapel unit tests are executed as follows:

```bash
make test-all
```

</details>

For more details regarding Arkouda testing, please consult the Python test [README](tests/README.md) and Chapel test
[README](test/README.md), respectively.


<a id="install-ak"></a>
## Installing the Arkouda Python Library and Dependencies <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
Now that the arkouda_server is built and tested, install the Python library.

The Arkouda Python library along with its dependent libraries are installed with pip. There are four types of 
Python dependencies for the Arkouda developer to install: requires, dev, test, and doc. The required libraries, 
which are the runtime dependencies of the Arkouda python library, are installed as follows:

```bash
 pip3 install -e .
```

Arkouda and the Python libraries required for development, test, and doc generation activities are installed
as follows:

```bash
pip3 install -e .[dev]
```

Alternatively you can build a distributable package via
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

<a id="run-ak"></a>
## Running arkouda\_server <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

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
Memory tracking is turned on by default now, you can run server with memory tracking turned off by

```bash
./arkouda_server --memTrack=false
```

By default, the server listens on port `5555`. This value can be overridden with the command-line flag 
`--ServerPort=1234`

Memory tracking is turned on by default and turned off by using the  `--memTrack=false` flag

Trace logging messages are turned on by default and turned off by using the `--trace=false` flag

Other command line options are available and can be viewed by using the `--help` flag

```bash
./arkouda-server --help
```

<a id="run-ak-sanity"></a>
### Sanity check arkouda\_server <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

To sanity check the arkouda server, you can run

```bash
make check
```

This will start the server, run a few computations, and shut the server down. In addition, the check script can be executed 
against a running server by running the following Python command:

```bash
python3 tests/check.py localhost 5555
```

<a id="run-ak-token-auth"></a>
### Token-Based Authentication in Arkouda <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

Arkouda features a token-based authentication mechanism analogous to Jupyter, where a randomized alphanumeric string is
generated or loaded at arkouda\_server startup. The command to start arkouda\_server with token authentication is as follows:

```bash
./arkouda_server --authenticate
```

The generated token is saved to the tokens.txt file which is contained in the .arkouda directory located in the same 
working directory the arkouda\_server is launched from. The arkouda\_server will re-use the same token until the 
.arkouda/tokens.txt file is removed, which forces arkouda\_server to generate a new token and corresponding
tokens.txt file.

In situations where a user-specified token string is preferred, this can be specified in the ARKOUDA_SERVER_TOKEN environment variable. As is the case with an Arkouda-generated token, the user-supplied token
is saved to the .arkouda/tokens.txt file for re-use.

<a id="run-ak-connect"></a>
### Connecting to Arkouda <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

The client connects to the arkouda\_server either by supplying a host and port or by providing a connect\_url connect string:

```bash
arkouda.connect(server='localhost', port=5555)
arkouda.connect(connect_url='tcp://localhost:5555')
```

When arkouda\_server is launched in authentication-enabled mode, clients connect by either specifying the access\_token
parameter or by adding the token to the end of the connect\_url connect string:

```bash
arkouda.connect(server='localhost', port=5555, access_token='dcxCQntDQllquOsBNjBp99Pu7r3wDJn')
arkouda.connect(connect_url='tcp://localhost:5555?token=dcxCQntDQllquOsBNjBp99Pu7r3wDJn')
```

Note: once a client has successfully connected to an authentication-enabled arkouda\_server, the token is cached in the
user's $ARKOUDA\_HOME .arkouda/tokens.txt file. As long as the arkouda_server token remains the same, the user can
connect without specifying the token via the access_token parameter or token url argument.


<a id="log-ak"></a>
## Logging <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

The Arkouda server features a Chapel logging framework that prints out the module name, function name and line number
for all logged messages. An example is shown below:

```
2021-04-15:06:22:59 [ConcatenateMsg] concatenateMsg Line 193 DEBUG [Chapel] creating pdarray id_4 of type Int64
2021-04-15:06:22:59 [ServerConfig] overMemLimit Line 175 INFO [Chapel] memory high watermark = 44720 memory limit = 30923764531
2021-04-15:06:22:59 [MultiTypeSymbolTable] addEntry Line 127 DEBUG [Chapel] adding symbol: id_4 
```

Available logging levels are ERROR, CRITICAL, WARN, INFO, and DEBUG. The default logging level is INFO where all messages at the ERROR, CRITICAL, WARN, and INFO levels are printed. The log level can be set globally by passing in the --logLevel parameter upon arkouda\_server startup. For example, passing the --logLevel=LogLevel.DEBUG parameter as shown below sets the global log level to DEBUG:

```
./arkouda_server --logLevel=LogLevel.DEBUG
```

In addition to setting the global logging level, the logging level for individual Arkouda modules can also be configured. For example, to set MsgProcessing to DEBUG for the purposes of debugging Arkouda array creation, pass the MsgProcessing.logLevel=LogLevel.DEBUG parameter upon arkouda\_server startup as shown below:

```
./arkouda_server --MsgProcessing.logLevel=LogLevel.DEBUG --logLevel=LogLevel.WARN
```

In this example, the logging level for all other Arkouda modules will be set to the global value WARN.

<a id="typecheck-ak"></a>
## Type Checking in Arkouda <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

Both static and runtime type checking are becoming increasingly popular in Python, especially for large Python code bases 
such as those found at [dropbox](https://dropbox.tech/application/our-journey-to-type-checking-4-million-lines-of-python). 
Arkouda uses [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking and [typeguard](https://typeguard.readthedocs.io/en/latest/) 
for runtime type checking.

<details>
 <summary><b>(click to see more)</b></summary>

Enabling runtime as well as static type checking in Python starts with adding [type hints](https://www.python.org/dev/peps/pep-0484/), 
as shown below to a method signature:

```
def connect(server : str="localhost", port : int=5555, timeout : int=0, 
                           access_token : str=None, connect_url=None) -> None:
```

mypy static type checking can be invoked either directly via the mypy command or via make:

```
$ mypy arkouda
Success: no issues found in 16 source files
$ make mypy
python3 -m mypy arkouda
Success: no issues found in 16 source files
```

Runtime type checking is enabled at the Python method level by annotating the method if interest with the @typechecked decorator, an 
example of which is shown below:

```
@typechecked
def save(self, prefix_path : str, dataset : str='array', mode : str='truncate') -> str:
```

Type checking in Arkouda is implemented on an "opt-in" basis. Accordingly, Arkouda continues to support [duck typing](https://en.wikipedia.org/wiki/Duck_typing) for parts of the Arkouda API where type checking is too confining to be useful. As detailed above, both runtime and static 
type checking require type hints. Consequently, to opt-out of type checking, simply leave type hints out of any method declarations where duck typing is desired.

</details>

<a id="env-vars-ak"></a>
## Environment Variables <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
The various Arkouda aspects (compilation, run-time, client, tests, etc.) can be configured using a number of environment
variables (env vars).  See the [ENVIRONMENT](ENVIRONMENT.md) documentation for more details.

<a id="versioning-ak"></a>
## Versioning <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
Beginning after tag `v2019.12.10` versioning is now performed using [Versioneer](https://github.com/python-versioneer/python-versioneer)
which determines the version based on the location in `git`.

An example using a hypothetical tag `1.2.3.4`
```bash
git checkout 1.2.3.4
python -m arkouda |tail -n 2
>> Client Version: 1.2.3.4
>> 1.2.3.4

# If you were to make uncommitted changes and repeat the command you might see something like:
python -m arkouda|tail -n 2
>> Client Version: 1.2.3.4+0.g9dca4c8.dirty
>> 1.2.3.4+0.g9dca4c8.dirty

# If you commit those changes you would see something like
python -m arkouda|tail -n 2
>> Client Version: 1.2.3.4+1.g9dca4c8
>> 1.2.3.4+1.g9dca4c8
```
In the hypothetical cases above _Versioneer_ tells you the version and how far / how many commits beyond the tag your repo is.

When building the server-side code the same versioning information is included in the build.  If the server and client do not
match you will receive a warning.  For developers this is a useful reminder when you switch branches and forget to rebuild.

```bash
# Starting the arkouda when built from tag 1.2.3.4 shows the following in the startup banner 
arkouda server version = 1.2.3.4

# If you built from an arbitrary branch the version string is based on the derived coordinates from the "closest" tag
arkouda server version = v2019.12.10+1679.abc2f48a

# The .dirty extension denotes a build from uncommitted changes, or a "dirty branch" in git vernacular
arkouda server version = v2019.12.10+1679.abc2f48a.dirty
```

For maintainers, creating a new version is as simple as creating a tag in the repository; i.e.
```bash
git checkout master
git tag 1.2.3.4
python -m arkouda |tail -n 2
>> Client Version: 1.2.3.4
>> 1.2.3.4
git push --tags
```

<a id="contrib-ak"></a>
## Contributing to Arkouda <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

If you'd like to contribute, please see [CONTRIBUTING.md](CONTRIBUTING.md).
