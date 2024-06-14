use IO;
use CTypes;

require "../src/parquet/WriteParquet.h";
require "../src/WriteParquet.o";
require "../src/parquet/ReadParquet.h";
require "../src/ReadParquet.o";
require "../src/parquet/UtilParquet.h";
require "../src/UtilParquet.o";

proc getVersionInfo() {
  extern proc c_getVersionInfo(): c_ptrConst(c_char);
  extern proc strlen(str): c_int;
  extern proc c_free_string(ptr);
  var cVersionString = c_getVersionInfo();
  defer {
    c_free_string(cVersionString: c_ptr(void));
  }
  var ret: string;
  try {
    ret = string.createCopyingBuffer(cVersionString,
                              strlen(cVersionString));
  } catch e {
    ret = "Error converting Arrow version message to Chapel string";
  }
  return ret;
}

proc main() {
  var ArrowVersion = getVersionInfo();
  writeln("Found Arrow version: ", ArrowVersion);
  return 0;
}
