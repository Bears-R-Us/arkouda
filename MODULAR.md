# Modular building of Arkouda

The modules that are included in each build of Arkouda can be found in the [ServerModules.cfg](ServerModules.cfg) file.

If only a certain subset of the Arkouda funcionality is required, modules can be commented out using a `#` prior to the module name in [ServerModules.cfg](ServerModules.cfg). This will exclude those modules from the next build of Arkouda, which can have significant improvements on build times, which can be particularly valuable during development when only a single feature is desired to be tested.

### Specifying a custom configuration file

In the case of having multiple configurations frequently switched between, the default `ServerModules.cfg` file can be replaced by setting the environment variable `ARKOUDA_CONFIG_FILE`.

New config files must follow the same format, listing one module name per line with comments having a `#` as the first character of the line.

### Adding new modules into the build process

Adding a module from outside of the Arkouda `src/` directory requires adding the module name in the `ServerModules.cfg` file, but also requires setting the `ARKOUDA_SERVER_USER_MODULES` environment variable to a string of the absolute path to the Chapel module. If multiple user modules are included, the paths must be separated by a space. For example, Arkouda could be built with a custom module `TestMsg.chpl` by adding `TestMsg` to a line in the `ServerModules.cfg` file and then by running the command `ARKOUDA_SERVER_USER_MODULES='/Users/path/to/TestMsg.chpl' make` in the Arkouda home directory. Note that for this to work, the absolute path to the `.chpl` file must be specified in the `ARKOUDA_SERVER_USER_MODULES` environment variable.

Additionally, a `registerMe()` function is required in the module in order to make the new functionality visible to the Arkouda server. This function must have the `CommandMap` module in scope and must call `registerFunction()` with the server message string and function name to be called.

Here is an example `registerFunction()` call taken from [src/KEXtremeMsg.chpl](src/KExtremeMsg.chpl):
```
proc registerMe() {
  use CommandMap;
  registerFunction("mink", minkMsg);
  registerFunction("maxk", maxkMsg);
}
```

The last step on adding a new function is to add a function for the client side of the server. This can be accomplished in a number of ways, but a simple approach is to create a script of this form:
```
import arkouda as ak
from arkouda.client import generic_msg

def test_command():
    rep_msg = generic_msg(cmd='test-command')

ak.__dict__["test_command"] = test_command
```
With the key elements being (1) the function calls `generic_msg()`, which will execute the command string specified by the argument `cmd` and (2) the function is added to the ak dictionary, enabling it to be called as `ak.test_command()`.

### Saving modules used in an Arkouda server run

When testing a specific workflow, it can sometimes be difficult to determine which modules are needed, but doing so can allow you to save significant amounts of build time. To help with this process of discovering which modules your workflow needs, the `--saveUsedModules` flag has been added to the Arkouda server.

To use this flag:
1. run your server with the flag: `./arkouda_server --saveUsedModules`
2. execute all commands that are needed for this particular workflow
3. shut down the server with `ak.shutdown()`

Upon server shutdown, a `UsedModules.cfg` file is created that includes the list of modules that were used in that particular Arkouda server instance. This can then either be inspected to assist in modification of the `ServerModules.cfg` file, or set to be used explicitly by the server by setting the `ARKOUDA_CONFIG_FILE` environment variable to this new file.

If you wish to determine what modules are used in a particular benchmark run, that can be done by running an Arkouda benchmark with the `--server-args` argument set: `./benchmarks/run-benchmarks.py BENCHMARK-NAME --server-args='--saveUsedModules'`.