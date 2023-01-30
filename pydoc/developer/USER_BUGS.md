.. _user_bugs:

Reproducing User Bugs Efficiently
=================================


This section contains suggestions for effectively reproducing user-reported bugs in combination with the `--saveUsedModules` flag to efficiently figure out which modules are required for testing changes for a specific bug to minimize build times.

~~~~~~~TODO: LINK TO OTHER PAGE~~~~~~~~~~~~

Now that you've got your one-time Arkouda release built (from the previous section), to reproduce a reported bug:
1. `./<ak-server-path/arkouda_server --saveUsedModules`
    - run the executable from the location that you saved it to from the previous step
    - don't forget the `--saveUsedModules` flag
3. reproduce the bug
4. shutdown the server with `ak.shutdown()` (to generate the `UsedModules.cfg` file)
5. `mv UsedModules.cfg <path-to-arkouda>/ServerModules.cfg` or other technique mentioned above
7. now, the server can be built and changes can be tested without compiling any unnecessary modules, since the `UsedModules.cfg` file will have only the minimal modules
    - note that the `UsedModules.cfg` will be created in the directory that the command to run the server is executed from

