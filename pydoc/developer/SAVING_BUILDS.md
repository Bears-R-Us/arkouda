.. _saving_builds:

Saving Full Builds
==================

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

Now, we have a full build of the release, so there shouldn't be any reason to need to rebuild the full server again until the next release. What I would recommend is doing a full `gasnet` build of the server overnight so you don't have to be impacted by the long build times and memory hogging during development. This _will_ take a while, especially for a gasnet build, but you should only have to do it once per release (and it can be done overnight).
