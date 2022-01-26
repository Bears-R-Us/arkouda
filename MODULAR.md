# Modular building of Arkouda

The modules that are included in each build of Arkouda can be found in the [ServerModules.cfg](ServerModules.cfg) file.

If only a certain subset of the Arkouda funcionality is required, modules can be commented out using a `#` prior to the module name in [ServerModules.cfg](ServerModules.cfg). This will exclude those modules from the next build of Arkouda, which can have significant improvements on build times, which can be particularly valuable during development when only a single feature is desired to be tested.

### Specifying a custom configuration file

In the case of having multiple configurations frequently switched between, the default `ServerModules.cfg` file can be replaced by setting the environment variable `ARKOUDA_CONFIG_FILE`.

New config files must follow the same format, listing one module name per line with comments having a `#` as the first character of the line.

### Adding new modules into the build process

The only unique aspect of adding modules from other directories into the Arkouda build process is that a complete path must be specified for all modules not found in the Arkouda `src/` directory. This is done by adding a line such as `/Users/test-user/path/to/mod` to `ServerModules.cfg` to add the module `mod.chpl` to the Arkouda build. Aside from that, adding a custom module into the Arkouda build process is treated the same way as a regular module: added in a single line in the server configuration file.

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