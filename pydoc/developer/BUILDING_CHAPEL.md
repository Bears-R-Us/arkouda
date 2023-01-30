.. _building_chapel:

Building Chapel
===============

Chapel can either be installed via brew or built from source.
Building Chapel from source allows developers to customize their
environment and check out specific versions of Chapel if needed.

The simplest way to build Chapel from source for Arkouda
developers that will work in most cases follows:

1. Clone the Chapel repo from https://github.com/chapel-lang/chapel.git
2. `git checkout release/1.29` (or whatever the latest release is)
3. `source util/setchplenv.bash` - set up your Chapel environment
4. Ensure that `CHPL_DEVELOPER` is unset
    - this flag gives the compiler some additional debug output and should only be used to build Chapel when developing in the compiler (which is infrequently for an Arkouda developer)
5. `make -j 8` - build the Chapel compiler in parallel

If any issues are encountered during these steps, see https://chapel-lang.org/docs/usingchapel/chplenv.htmlv and https://chapel-lang.org/docs/usingchapel/building.html for more information.
