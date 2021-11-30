use Parquet, SysCTypes, CPtr, FileSystem;
use UnitTest;
use TestBase;

proc testReadWrite(filename: c_string, dsetname: c_string, size: int) {
  extern proc c_readColumnByName(filename, chpl_arr, colNum, numElems);
  extern proc c_writeColumnToParquet(filename, chpl_arr, colnum,
                                     dsetname, numelems, rowGroupSize);
  var a: [0..#size] int;
  for i in 0..#size do a[i] = i;
  c_writeColumnToParquet(filename, c_ptrTo(a), 0, dsetname, size, 10000);

  var b: [0..#size] int;
  
  c_readColumnByName(filename, c_ptrTo(b), dsetname, size);
  if a.equals(b) {
    return 0;
  } else {
    writeln("FAILED: read/write");
    return 1;
  }
}

proc testGetNumRows(filename: c_string, expectedSize: int) {
  extern proc c_getNumRows(chpl_str): int;
  var size = c_getNumRows(filename);
  if size == expectedSize {
    return 0;
  } else {
    writeln("FAILED: c_getNumRows");
    return 1;
  }
}

proc testGetType(filename: c_string, dsetname: c_string) {
  extern proc c_getType(filename, colname): c_int;
  var arrowType = c_getType(filename, dsetname);

  // a positive value corresponds to an arrow type
  // -1 corresponds to unsupported type
  if arrowType >= 0 {
    return 0;
  } else {
    writeln("FAILED: c_getType with ", arrowType);
    return 1;
  }
}

proc testVersionInfo() {
  extern proc c_getVersionInfo(): c_string;
  extern proc c_free_string(ptr);
  var cVersionString = c_getVersionInfo();
  defer {
    c_free_string(cVersionString);
  }
  var ret;
  try! ret = createStringWithNewBuffer(cVersionString);
  if ret[0]: int >= 5 {
    return 0;
  } else {
    return 1;
  }
}
proc main() {
  var errors = 0;

  const size = 1000;
  const filename = "myFile.parquet".c_str();
  const dsetname = "my-dset-name-test".c_str();
  
  errors += testReadWrite(filename, dsetname, size);
  errors += testGetNumRows(filename, size);
  errors += testGetType(filename, dsetname);
  errors += testVersionInfo();

  if errors != 0 then
    writeln(errors, " Parquet tests failed");

  remove("myFile.parquet");
}
