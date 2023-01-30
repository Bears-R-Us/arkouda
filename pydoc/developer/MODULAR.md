.. _modular_builds:

Using the Modular Build System
===============================

If you are looking to speed up build times, this section contains information for using the modular build system. This system can significantly decrease compilation times by reducing the number of features built into the server, focusing on just the crucial ones. In general, developers should always be using the modular build system when working on server-side development.

One of the most effective ways to reduce compile times is to only bundle the Arkouda modules you need into your server—basically don't compile code that you aren't developing or testing at the moment. The following is a technique to determine which modules those might be rather than trying to guess or reason through it manually.

If you already have a full-server build:
1. `./arkouda_server --saveUsedModules` - run server with used module tracking
4. run the piece of code that you will be modifying
    - e.g., for the stream benchmark, you could run that benchmark with `python3 benchmarks/stream.py <hostname> <port>`
    - for something like a new function you are testing, you could just connect to the server and run that single function interactively
9. if you ran a script (like the stream benchmark) then run an interactive session and connect to the server (if you were just executing code interactively, you should already have an interactive session connected)
10. run `ak.shutdown()` from the interactive session
    - the Arkouda server _MUST_ be properly shutdown to generate the `UsedModules.cfg` file
    - a `Ctrl-c` or equivalent will not generate the file
11. now you have a `UsedModules.cfg` file in the top-level Arkouda directory that lists all of the modules required to run that piece of code
    - easiest way to use that is by running `mv UsedModules.cfg ServerModules.cfg` from the Arkouda top-level directory
    - can also be specified with the `ARKOUDA_CONFIG_FILE` env variable to override the `ServerModules.cfg` file

Alternatively, if you are doing something like just adding a new module or know what functions you'll need, you can just modify the `ServerModules.cfg` file and comment out with a `#` or just remove the lines of the modules you don't need.

This is the single biggest way to save on compilation time, but is a bit more involved than the others. For reference, a full build on my machine takes ~500 seconds, but with minimal modules, it only takes ~30 seconds.