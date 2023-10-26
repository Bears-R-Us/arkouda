## Compiling
Compilation command: `g++ read-parquet.cpp -O3 -std=c++17 <include/link flags>`

The trick to figuring out how to compile on any system would be to run `make compile-arrow-cpp` in Arkouda and then cancel it once you can see the compilation command and steal the compilation flags from there. You will likely see `-I`, `-L`, and `-l` flags, depending on where you run and copy-pasting those onto the command above into the `<include/link flags>` section should do the trick