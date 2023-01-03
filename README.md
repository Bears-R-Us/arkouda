<p align="center">
  <img src="pictures/arkouda_wide_marker1.png"/>
</p>

<h2 align="center">Arkouda (αρκούδα) :bear:</br>Interactive Data Analytics at Supercomputing Scale</h2>

<p align="center">
<a href="https://github.com/Bears-R-Us/arkouda/actions/workflows/CI.yml"><img alt="Actions Status" src="https://github.com/Bears-R-Us/arkouda/workflows/CI/badge.svg"></a>
<a href="https://bears-r-us.github.io/arkouda/"><img alt="Documentation Status" src="https://github.com/Bears-R-Us/arkouda/workflows/docs/badge.svg"></a>
<a href="https://github.com/Bears-R-Us/arkouda/blob/master/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## Online Documentation
[Arkouda docs at Github Pages](https://bears-r-us.github.io/arkouda/)

## Nightly Arkouda Performance Charts
[Arkouda nightly performance charts](https://chapel-lang.org/perf/arkouda/)

## Gitter channels
[Arkouda Gitter channel](https://gitter.im/ArkoudaProject/community)

[Chapel Gitter channel](https://gitter.im/chapel-lang/chapel)

## Talks on Arkouda

[Mike Merrill's SIAM PP-22 Talk](https://chapel-lang.org/presentations/Arkouda_SIAM_PP-22.pdf)

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
provided that you run on a sufficiently capable compute server. Arkouda breaks 
the shared memory paradigm and scales its operations to dataframes with over
200 billion rows, maybe even a trillion. In practice we have run Arkouda server
operations on columns of one trillion elements running on 512 compute nodes.
This yielded a >20TB dataframe in Arkouda.

<a id="toc"></a>
# Table of Contents

1. [Prerequisites](#prereqs)
2. [Building Arkouda](#build-ak)
3. [Testing Arkouda](#test-ak)
4. [Running arkouda_server](#run-ak)
   - [Running the arkouda_server From a Script](#run-server-script)
   - [Sanity check](#run-ak-sanity)
   - [Token-Based Authentication](#run-ak-token-auth)
   - [Setting Per-Locale Memory and CPU Core Limits](#set-locale-memory-cpu-core-limits)
   - [Connecting to Arkouda](#run-ak-connect)
5. [Logging](#log-ak)
6. [Type Checking in Arkouda](#typecheck-ak)

7. [Environment Variables](#env-vars-ak)
8. [Versioning](#versioning-ak)
9. [External Systems Integration](#external-integration)
10. [Metrics](#metrics)
11. [Asynchronous Client](#async_client)
12. [Contributing](#contrib-ak)


<a id="prereqs"></a>
## Prerequisites <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

For a complete list of requirements for Arkouda, please review [REQUIREMENTS.md](pydoc/setup/REQUIREMENTS.md).

For detailed prerequisite information and installation guides, please review the install guide for your operating system.
- [Linux Install](pydoc/setup/LINUX_INSTALL.md)
- [MacOS Install](pydoc/setup/MAC_INSTALL.md)
- [Windows Install](pydoc/setup/WINDOWS_INSTALL.md)

<a id="build-ak"></a>
## Building Arkouda <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
In order to run the Arkouda server, it must first be compiled. Detailed instructions on the build process can be found at [BUILD.md](pydoc/setup/BUILD.md).

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

Trace logging messages are turned on by default and turned off by using the `--trace=false` flag

Other command line options are available and can be viewed by using the `--help` flag

```bash
./arkouda-server --help
```

<a id="run-server-script"></a>
### Running the arkouda_server From a Script

With the addition of two server startup flags, `--autoShutdown` and `--serverInfoNoSplash`, running the arkouda_server from a script is easier than ever.

To connect to the server via a script, you'll first have to issue a subprocess command to start the `arkouda_server` with the optional configuration flags.

```python
import subprocess

# Update the below path to point to your arkouda_server
cmd = "/Users/<username>/Documents/git/arkouda/arkouda_server -nl 1 --serverInfoNoSplash=true --autoShutdown=true"
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
```

This will allow you to access the server output, which on launch using `--serverInfoNoSplash=true` will be a JSON string with the server configuration which can be parsed for the server host, port, and other potentially useful information.

For a full example and explanation, view the [Running From Script](training/RUNNING_FROM_SCRIPT.md) document.

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

<a id="set-locale-memory-cpu-core-limits"></a>
### Setting Per-Locale Memory and CPU Core Limits <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

By default, each Arkouda locale utilizes all available memory and CPU cores on the host machine. However, it is possible to set per-locale limits for both memory as well as CPU cores. 

#### Per-Locale Memory Limits

There are three approaches to setting the max memory used by each Arkouda locale. Firstly, the built-in Chapel approach sets the max per-locale memory to an explicit number of bytes via the --memMax startup parameter. For example, to set the max memory utilized by each locale to 100 GB, the Arkouda startup command would include the following:

```
./arkouda_server --memMax=100000000000
```

The Arkouda dynamic memory limit approach sets the per-locale memory limit based upon a configurable percentage of available memory on each locale host. Prior to the execution of each command, the MemoryMgmt [localeMemAvailable](https://github.com/Bears-R-Us/arkouda/blob/e4a48c52eb00097e6e1dfa365cbc586e2e988a85/src/MemoryMgmt.chpl#L133) function does the following on each locale:

1. Verifies the projected, additional per-locale memory required by the incoming command does not exceed the memory currently allocated to Arkouda. If the projected, additional memory is within the memory currently allocated to Arkouda on each locale, the command is allowed to proceed.
2. If the projected, additional per-locale memory exceeds the memory currently allocated to Arkouda on any locale, localeMemAvailable checks if the configurable percentage of available memory on each node will accommodate the projected, additional memory of the incoming command. If so, the command is allowed to proceed.
3. If the projected, additional per-locale memory required by the incoming command exceeds the configured percentage of available memory on any locale, localeMemAvailable returns false and a corresponding error is [thrown](https://github.com/Bears-R-Us/arkouda/blob/e4a48c52eb00097e6e1dfa365cbc586e2e988a85/src/ServerConfig.chpl#L348) in the ServerConfig [overMemLimit](https://github.com/Bears-R-Us/arkouda/blob/e4a48c52eb00097e6e1dfa365cbc586e2e988a85/src/ServerConfig.chpl#L286) function. 

In the example below, dynamic memory checking is enabled with the default availableMemoryPct of 90, configuring Arkouda to throw an error if (1) the projected, additional memory required for a command exceeds memory currently allocated to Arkouda on 1..n locales and (2) the projected, additional memory will exceed 90 percent of available memory on 1..n locales. 

```
./arkouda_server --MemoryMgmt.memMgmtType=MemMgmtType.DYNAMIC
```

Setting additionalMemoryPct to 70 would result in the following startup command:

```
./arkouda_server --MemoryMgmt.memMgmtType=MemMgmtType.DYNAMIC ----MemoryMgmt.additionalMemoryPct=70
```

Important note: dynamic memory checking _works on Linux and Unix systems only._

In the final, default approach, the max memory utilized by each locale is set as percentage of physical memory on the locale0 host, defaulting to 90 percent. If another percentage is desired, this is set via the --perLocaleMemLimit startup parameter. For example, to set max memory utilized by each locale to seventy (70) percent of physical memory on locale0, the Arkouda startup command would include the following:

```
./arkouda_server --perLocaleMemLimit=70
```

#### Per-Locale CPU Core Limits

The max number of CPU cores utilized by each locale is set via the CHPL_RT_NUM_THREADS_PER_LOCALE environment variable. An example below sets the maximum number of cores for each locale to 16:

```
export CHPL_RT_NUM_THREADS_PER_LOCALE=16
```

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

### Log Levels

Available logging levels are ERROR, CRITICAL, WARN, INFO, and DEBUG. The default logging level is INFO where all messages at the ERROR, CRITICAL, WARN, and INFO levels are printed. The log level can be set globally by passing in the --logLevel parameter upon arkouda\_server startup. For example, passing the --logLevel=LogLevel.DEBUG parameter as shown below sets the global log level to DEBUG:

```
./arkouda_server --logLevel=LogLevel.DEBUG
```

In addition to setting the global logging level, the logging level for individual Arkouda modules can also be configured. For example, to set MsgProcessing to DEBUG for the purposes of debugging Arkouda array creation, pass the MsgProcessing.logLevel=LogLevel.DEBUG parameter upon arkouda\_server startup as shown below:

```
./arkouda_server --MsgProcessing.logLevel=LogLevel.DEBUG --logLevel=LogLevel.WARN
```

In this example, the logging level for all other Arkouda modules will be set to the global value WARN.

### Log Channels

Arkouda logs can be written either to the console (default) or to the arkouda.log file located in the .arkouda directory. To enable log output to the arkouda.log file, start Arkouda as follows with the --logChannel flag:

```
./arkouda_server --logChannel=LogChannel.FILE
```

### Arkouda Command Logging

All incoming Arkouda server commands submitted by the Arkouda client can be logged to the commands.log file located in the .arkouda directory. Arkouda command logging is enabled as follows:

```
./arkouda_server --logCommands=true
```

The Arkouda command logging capability has a variety of uses, one of which is replaying analytic or data processing scenarios in either interactive or batch mode. Moreover, a sequence of Arkouda server commands provides the possibility of utilizing Arkouda clients developed in other languages such as Rust or Go. In still another use case, command logging in Arkouda provides a command sequence for starting Arkouda via cron job and processing large amounts of data into Arkouda arrays or dataframes, thereby obviating the need for a user to wait for well-known data processing/analysis steps to complete; this use case is of particular value in situations where the data loading process is particularly time-intensive. Finally, command logging provides a means of integrating a non-interactive Arkouda data processing/analysis sequence into a data science workflow implemented in a framework such as Argo Workflows or Kubeflow.

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


<a id="versioning-ak"></a>
## Versioning <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
Beginning after tag `v2019.12.10` versioning is now performed using [Versioneer](https://github.com/python-versioneer/python-versioneer)
which determines the version based on the location in `git`.

An example using a hypothetical tag 1.2.3.4

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

<a id="external-integration"></a>
## External Systems Integration <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

Integrating Arkouda with cloud environments enables users to access Arkouda from machine learning (ML) and deep learning (DL) workflows deployed to Kubernetes as an example. Detailed discussions regarding Arkouda systems integration and specific instructions for registering/deregistering Arkouda with Kubernetes are located in [EXTERNAL INTEGRATION.md](EXTERNAL_INTEGRATION.md)

<a id="metrics"></a>
## Metrics <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

Arkouda provides a separate, dedicated zmq socket to enable generation and export of a variety of system, locale, user, and request metrics. Arkouda generated metrics in a format compatible with Prometheus, Grafana, and TimescaleDB. An Arkouda Prometheus exporter that serves as a Prometheus scrape target will be made available soon in the [arkouda-contrib](https://github.com/Bears-R-Us/arkouda-contrib) repository. A detailed discussion of Arkouda metrics is located in [METRICS.md](METRICS.md)

<a id="async_client"></a>
## Asynchronous Client

### Background

Arkouda has an alpha capability for enabling asynchronous client-server communications that provides feedback to users that a request has been submitted and is being processed within the Arkouda server. The initial asynchronous request capability supports multiuser Arkouda use cases where users may experience delays when the Arkouda server is processing requests by 1..n other users.

### Configuration

To enable asynchronous client communications, set the ARKOUDA_REQUEST_MODE environment variable as follows:

```
export ARKOUDA_REQUEST_MODE=ASYNC
```

### Exiting the Python shell

As of 01022023, exiting the Python shell in ASYNC request mode requires the following command:

```
ak.exit()
```

<a id="contrib-ak"></a>
## Contributing to Arkouda <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

If you'd like to contribute, we'd love to have you! Before jumping in and adding issues or writing code, please see [CONTRIBUTING.md](CONTRIBUTING.md).
