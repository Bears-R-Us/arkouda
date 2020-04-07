# Python Tests against the arkouda\_server

All Python tests can be run against a local or remote arkouda\_server. The arkouda project is configured so that
no special Python paths need to be defined.
  
# The arkouda Python test harness

The arkouda project has a [pytest](https://docs.pytest.org/en/latest/)-based Python test harness that can be executed 
via the command-line interface (cli) or within the make file. 

## pytest environment preparation

There are two python libraries that must be installed to execute the arkouda test harness, pytest and pytest-env,
both of which can be installed via pip:
```
pip install pytest, pytest-env
```
## Configuration of pytest: the pytest.ini file

The pytest.ini file configures the arkouda Python test harness, specifically the following parameters:

```
[pytest]
testpaths = 
    tests/base_test.py
    tests/client_test.py
    tests/compare_test.py
    tests/dtypes_tests.py
    tests/groupby_test.py
    tests/join_test.py
    tests/operator_tests.py
    tests/setops_test.py
    tests/string_test.py
    tests/where_test.py
norecursedirs = .git dist build *egg* tests/deprecated/*
python_functions = test*
env =
    TEST_DATA_URL=localhost
    ARKOUDA_SERVER_HOST=localhost
    ARKOUDA_SERVER_PORT=5555
    FULL_STACK_TEST=False
    VERBOSE=False
```
* testpaths: shows the paths to all test files. For the time-being, the arkouda unit tests to be executed are 
specified on a per-file basis, but pytest can also support directory-level configuration. 
* norecursedirs: indicates which directories contain files that should be ignored, including tests/deprecated
* python\_functions: the naming pattern for all test functions. Ih the case of arkouda, all python method (functions)
  with names starting with "test" are executed by pytest
* env: the pytest env variables needed to execute the arkouda Python test harness 

## arkouda pytest environmental variables
* TEST\_DATA\_URL: the url from which data files used in arkouda Python tests are access (local or remote)
* ARKOUDA\_SERVER\_HOST: the hostname or ip address where the arkouda\_server is located
* ARKOUDA\_SERVER\_PORT: the port the arkouda\_server is listening on
* FULL_STACK_TEST: if True, the TestCase.setUpClass method starts up an arkouda\_server on the local machine, where
  server=ARKOUDA\_SERVER\_HOST and port= ARKOUDA\_SERVER\_PORT. If False, the test harness runs in client mode and 
  consequently an arkouda\_server is not started up.


# Running the arkouda Python test harness

To execute all tests in the arkouda test harness via the command line, execute the following command:

```
pytest -c pytest.ini 
```

To execute all tests in the arkouda test harness via make, execute the following command:

```
make test-python
```
pytest also enables a subset of 1..n unit tests to be executed. An example is shown below:

```
pytest tests/client_test.py
```
# Executing arkouda Python tests outside the test harness

The arkouda test classes can also be executed within an IDE such as [PyCharm](https://www.jetbrains.com/pycharm/) or 
[Eclipse](https://www.eclipse.org/ide/), either in run or debug mode.