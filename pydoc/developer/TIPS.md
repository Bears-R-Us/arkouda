.. _build_tips:

Tips for Speeding up Arkouda Builds
===================================
If you are looking for quick, 1-step compilation improvements, start here.

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
