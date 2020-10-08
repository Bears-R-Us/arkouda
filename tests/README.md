# Arkouda Python Unit Tests

The arkouda project has a [pytest](https://docs.pytest.org/en/latest/)-based Python test harness that can be executed 
via the command-line interface (CLI) or within the make file. 

All Python tests can be run against a local or remote arkouda\_server. The arkouda project is configured so that
no special Python paths need to be defined.

## pytest environment preparation

There are two python libraries that must be installed to execute the arkouda test harness, pytest and pytest-env,
both of which can be installed via pip3:

```
pip3 install pytest, pytest-env
```
## Configuration of pytest: the pytest.ini file

The pytest.ini file configures the arkouda Python test harness, specifically the following parameters:

```
[pytest]
testpaths = 
    tests/client_test.py
    tests/compare_test.py
    tests/dtypes_tests.py
    tests/groupby_test.py
    tests/io_test.py
    tests/io_util_test.py
    tests/join_test.py
    tests/operator_tests.py
    tests/security_test.py
    tests/setops_test.py
    tests/string_test.py
    tests/where_test.py
norecursedirs = .git dist build *egg* tests/deprecated/*
python_functions = test*
env =
    D:ARKOUDA_SERVER_HOST=localhost
    D:ARKOUDA_SERVER_PORT=5555
    D:ARKOUDA_FULL_STACK_TEST=True
    D:ARKOUDA_NUMLOCALES=2
    D:ARKOUDA_VERBOSE=True
    D:ARKOUDA_CLIENT_TIMEOUT=0
    D:ARKOUDA_LOG_LEVEL=INFO
```
* testpaths: shows the paths to all test files. For the time-being, the arkouda unit tests to be executed are 
specified on a per-file basis, but pytest can also support directory-level configuration.
* norecursedirs: indicates which directories contain files that should be ignored, including tests/deprecated
* python\_functions: the naming pattern for all test functions. Ih the case of arkouda, all python method (functions)
  with names starting with "test" are executed by pytest
* env: the pytest env variables needed to execute the arkouda Python test harness 

## arkouda pytest environmental variables

* ARKOUDA\_SERVER\_HOST: the hostname or ip address where the arkouda\_server is located. Defaults to localhost
* ARKOUDA\_SERVER\_PORT: the port the arkouda\_server is listening on. Defaults to 5555
* ARKOUDA\_FULL\_STACK\_TEST: if True, the TestCase.setUpClass method starts up an arkouda\_server on the local machine, where
  server=ARKOUDA\_SERVER\_HOST and port= ARKOUDA\_SERVER\_PORT. If False, the test harness runs in client mode and 
  consequently an arkouda\_server is not started up. Defaults to True
* ARKOUDA\_NUMLOCALES: sets number of locales if arkouda\_server is built with multilocale support
* ARKOUDA\_VERBOSE: if True, logging is set to DEBUG. Defaults to False
* ARKOUDA\_CLIENT\_TIMEOUT: the connection timeout for arkouda client. Defaults to 10 seconds
* ARKOUDA\_LOG\_LEVEL: the ArkoudaLogger level, can be DEBUG, INFO, WARNING, ERROR, or CRITICAL, defaults to INFO

NOTE: the Arkouda pytest env variables can be set within the pytest.ini file as above or in .bashrc or .bash_profile

# Running the arkouda Python test harness

To execute all tests in the arkouda test harness via the command line, execute the following command. Note the specification of
the python3 binary, since Arkouda requires Python3.6+.

```
python3 -m pytest -c pytest.ini 

# execute tests with logging enabled 
python3 -m pytest -c pytest.ini -s

# execute tests, exiting if one test case fails
python3 -m pytest -c pytest.ini -x
```

pytest also enables a subset of 1..n unit tests to be executed. An example is shown below:

```
# as above, -s outputs all print statements, -x exits if one test case fails
python3 -m pytest tests/client_test.py
```

To execute all Python tests in the arkouda Python test harness via make, execute one the following commands:

```
make test-python
make test
```

# Executing arkouda Python tests outside the test harness

The Arkouda test classes can also be executed within an IDE such as [PyCharm](https://www.jetbrains.com/pycharm/) or 
[Eclipse](https://www.eclipse.org/ide/), either in run or debug mode.

To run all Python and Chapel tests, run make test-all:

```
make test-all
```
