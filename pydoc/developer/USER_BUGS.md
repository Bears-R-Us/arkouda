Tips for Reproducing User Bugs
==============================

## Saving Full Builds

This section contains suggestions for cutting down the number of full module and/or gasnet builds that are needed for debugging.

When a user reports a bug, that typically comes from the previous release. It can be pretty frustrating to have to go back and rebuild the entire server for the past release just to reproduce that bug that you don't have a good idea what modules it needs.

To avoid having to recompile the release to reproduce a bug, here is what I like to do:

First, when an Arkouda release comes out, check out that tag and build the full server:
1. `git fetch --tags --all` - get new tags from github
3. `git tag` - list the tags that you have available
5. `git checkout <release-tag>` (e.g., `git checkout v2023.01.11`)
6. `make` - build the server
7. `mv arkouda_server <ak-server-path>`
    - here, we are moving that full server executable to a different place so it sticks around and isn't deleted after doing a different build of Arkouda
    - I just move mine up a single directory (so, `mv arkouda_server ../`, but put it wherever you'll remember and it won't be deleted)

Now, we have a full build of the release, so there shouldn't be any reason to need to rebuild the full server again until the next release. What I would recommend is doing a full [`gasnet` build of the server](GASNET.md) overnight so you don't have to be impacted by the long build times and memory hogging during development. This _will_ take a while, especially for a gasnet build, but you should only have to do it once per release (and it can be done overnight).


## Reproducing User Bugs Efficiently

This section contains suggestions for effectively reproducing user-reported bugs in combination with the `--saveUsedModules` flag to efficiently figure out which modules are required for testing changes for a specific bug to minimize build times. See the [modular build documentation](../setup/MODULAR.md) for more information on the `--saveUsedModules` flag.

Now that you've got your one-time Arkouda release built (from the previous section), to reproduce a reported bug:
1. `./<ak-server-path/arkouda_server --saveUsedModules`
    - run the executable from the location that you saved it to from the previous step
    - don't forget the `--saveUsedModules` flag
3. reproduce the bug
4. shutdown the server with `ak.shutdown()` (to generate the `UsedModules.cfg` file)
5. `mv UsedModules.cfg <path-to-arkouda>/ServerModules.cfg` or other technique mentioned above
7. now, the server can be built and changes can be tested without compiling any unnecessary modules, since the `UsedModules.cfg` file will have only the minimal modules
    - note that the `UsedModules.cfg` will be created in the directory that the command to run the server is executed from

