# Contributing to Arkouda

Arkouda welcomes contributions via feedback, bug reports, and pull requests.

## Development for Arkouda

### Makefile

Run `make help` to see a list of valid targets for `make <target>`.

The Makefile has several categories of targets.
* `default` - builds `arkouda_server`
* `archive` - archives this Git repository's *local* master branch
* `doc` - generates documentation
* `test` - builds server test binaries in directory `test-bin`
* `clean` - cleans build products

### Running the Test Suite

To build all test binaries for the backend Arkouda server, run `make -j test`.

Due to long build times, it is typical to build specific tests as needed. Run
`make test-help` for a list of available test binaries.

```terminal
make -j test-bin/Test1 test-bin/Test2
```

The test targets will *only* build the test binaries. You are required to run
the tests manually at this moment.

To see `chpl` debugging output, run `make` with `VERBOSE=1`.

```terminal
make VERBOSE=1 test-bin/Test1
```

Parallel `make -j` is supported, but the output will be interleaved, so
`VERBOSE` is not particularly useful with `make -j`.

While not recommended, if you want to fully specify flags to `chpl` when
building the test binaries, you can set `TEST_CHPL_FLAGS`.

```terminal
make -j test TEST_CHPL_FLAGS="--fast"
```

## Loosely Specified Coding Conventions

### Python3

 * `lowerCamelCase` for variable names
```python
printThreshold = 100
```

 * `names_with_underscores` for functions
```python
def print_it(x):
    print(x)
```

### Chapel

 * `lowerCamelCase` for variable names and procedures
```chapel
var aX: [{0..#s}] real;
proc printIt(x) {
     writeln(x);
}
```

 * `UpperCamelCase` for class names
```chapel
class Foo: FooParent {}
```
