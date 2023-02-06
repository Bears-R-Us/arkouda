Speeding up Arkouda Compilation
===============================

## Environment Variables to _Always_ Set

This section has a couple of environment variables that should _always_ be set/unset if you are looking for compilation time improvements.

1. `export ARKOUDA_QUICK_COMPILE=1`
    - this env variable turns off a couple of optimizations, speeding up builds
    - performance will be worse when compiled with this env variable, but that typically does not matter when doing development/correctness tests
7. `unset ARKOUDA_DEVELOPER`
    - this env variable should be unset to get faster builds because it currently overrides `ARKOUDA_QUICK_COMPILE`, so if you have both set, you will not get the quick compile speed up 
    - this env variable enables checks in the Chapel generated code, so performance takes a bit of a hit
8. `unset CHPL_DEVELOPER`
     - if this was set at the time that the chpl compiler was set, then the chpl compiler should be rebuilt with this unset
    - if you are using a homebrew installation of Chapel, you shouldn't need to worry about rebuilding the chpl compiler

## Using the Modular Build System

This is the single biggest way to save on compilation time, but is a bit more involved than the others. For reference, a full build on my machine takes ~500 seconds, but with minimal modules, it only takes ~30 seconds.

If you are looking to speed up build times, this section contains information for using the modular build system. This system can significantly decrease compilation times by reducing the number of features built into the server, focusing on just the crucial ones. In general, developers should always be using the modular build system when working on server-side development.

If you already have a full-server build:
1. `./arkouda_server --saveUsedModules` - run server with used module tracking
2. run the piece of code that you will be modifying
    - e.g., for the stream benchmark, you could run that benchmark with `python3 benchmarks/stream.py <hostname> <port>`
    - for something like a new function you are testing, you could just connect to the server and run that single function interactively
3. if you ran a script (like the stream benchmark) then run an interactive session and connect to the server (if you were just executing code interactively, you should already have an interactive session connected)
4. run `ak.shutdown()` from the interactive session
    - the Arkouda server _MUST_ be properly shutdown to generate the `UsedModules.cfg` file
    - a `Ctrl-c` or equivalent will not generate the file
5. now you have a `UsedModules.cfg` file in the top-level Arkouda directory that lists all of the modules required to run that piece of code
    - easiest way to use that is by running `mv UsedModules.cfg ServerModules.cfg` from the Arkouda top-level directory
    - can also be specified with the `ARKOUDA_CONFIG_FILE` env variable to override the `ServerModules.cfg` file

Alternatively, if you are doing something like just adding a new module or know what functions you'll need, you can just modify the `ServerModules.cfg` file and comment out with a `#` or just remove the lines of the modules you don't need.

For more general information on using the modular build system, please see the documentation [here](../setup/MODULAR.md).
