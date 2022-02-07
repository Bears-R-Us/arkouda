use SysCTypes, CPtr, IO;

require "../src/ArrowFunctions.h";
require "../src/ArrowFunctions.o";

proc getVersionInfo() {
  extern proc c_getVersionInfo(): c_string;
  extern proc strlen(str): c_int;
  extern proc c_free_string(ptr);
  var cVersionString = c_getVersionInfo();
  defer {
    c_free_string(cVersionString: c_void_ptr);
  }
  var ret: string;
  try {
    ret = createStringWithNewBuffer(cVersionString,
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
