use Parquet;

proc main() {
  // Reference a symbol from the external `Parquet` Mason package that is defined
  // in its Arrow-backed C++ prerequisites. This forces the program to compile
  // and link against Arrow/Parquet, verifying the dependency is installed and
  // the package builds correctly.
  writeln("Found Arrow-backed Parquet package (ARROWINT64=", ARROWINT64, ")");
  return 0;
}
