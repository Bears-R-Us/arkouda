Reducing Memory Usage of Arkouda Builds
=======================================

If your builds have been taking up too much memory, this section provides a method to reduce the memory usage by using the C backend and building the server in parallel.

This feature uses the C-backend to leverage incremental building and avoid building everything at once, which can cut down on the memory usage of the compiler.

This feature can also speed up build times, but is somewhat of a heroic step, as it requires using Chapel's (non-default) C back-end and will also negatively impact Arkouda execution performance. It can also only accelerate the `makeBinary` step of compilation, so is most useful in cases where that is a bottleneck in your compilation timings (where, in our full-scale Arkouda compiles, resolve tends to be the current bottleneck).

1. `make -j 8 runtime "CHPL_TARGET_COMPILER=clang"` (or gnu)
    - rebuild the Chapel runtime using the C backend
2. `make CHPL_FLAGS='--target-compiler=clang -j16'` (or gnu)
    - compile Arkouda in parallel

