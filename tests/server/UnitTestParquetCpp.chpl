use ParquetMsg, CTypes, FileSystem;
use UnitTest;
use TestBase;

proc testReadWrite(filename: c_string, dsetname: c_string, size: int) {
  extern proc c_readColumnByName(filename, chpl_arr, colNum, numElems, startIdx, batchSize, errMsg): int;
  extern proc c_writeColumnToParquet(filename, chpl_arr, colnum,
                                     dsetname, numelems, rowGroupSize, compressed,
                                     dtype, errMsg): int;
  extern proc c_free_string(a);
  extern proc strlen(a): int;
  var errMsg: c_ptr(uint(8));
  defer {
    c_free_string(errMsg);
  }
  var causeError = "cause-error":c_string;
  
  var a: [0..#size] int;
  for i in 0..#size do a[i] = i;

  if c_writeColumnToParquet(filename, c_ptrTo(a), 0, dsetname, size, 10000, false, 1, errMsg) < 0 {
    var chplMsg;
    try! chplMsg = string.createCopyingBuffer(errMsg, strlen(errMsg));
    writeln(chplMsg);
  }

  var b: [0..#size] int;

  if(c_readColumnByName(filename, c_ptrTo(b), dsetname, size, 0, 10000, c_ptrTo(errMsg)) < 0) {
    var chplMsg;
    try! chplMsg = string.createCopyingBuffer(errMsg, strlen(errMsg));
    writeln(chplMsg);
  }
    
  if a.equals(b) {
    return 0;
  } else {
    writeln("FAILED: read/write");
    return 1;
  }
}

proc testInt32Read() {
  extern proc c_readColumnByName(filename, chpl_arr, colNum, numElems, startIdx, batchSize, errMsg): int;
  extern proc c_free_string(a);
  extern proc strlen(a): int;
  var errMsg: c_ptr(uint(8));
  defer {
    c_free_string(errMsg);
  }
  
  var a: [0..#50] int;
  var expected: [0..#50] int;
  for i in 0..#50 do expected[i] = i;
  
  if(c_readColumnByName("resources/int32.parquet".c_str(), c_ptrTo(a),
                        "array".c_str(), 50, 0, 1, c_ptrTo(errMsg)) < 0) {
    var chplMsg;
    try! chplMsg = string.createCopyingBuffer(errMsg, strlen(errMsg));
    writeln(chplMsg);
  }

  if a.equals(expected) then
    return 0;
  else {
    writeln("FAILED: int32 read");
    return 1;
  }
}

proc testGetNumRows(filename: c_string, expectedSize: int) {
  extern proc c_getNumRows(chpl_str, err): int;
  extern proc c_free_string(a);
  extern proc strlen(a): int;
  var errMsg: c_ptr(uint(8));
  defer {
    c_free_string(errMsg);
  }
  var causeError = "asdasdasd":c_string;
  var size = c_getNumRows(filename, c_ptrTo(errMsg));
  if size == expectedSize {
    return 0;
  } else {
    var chplMsg;
    try! chplMsg = string.createCopyingBuffer(errMsg, strlen(errMsg));
    writeln(chplMsg);
    writeln("FAILED: c_getNumRows");
    return 1;
  }
}

proc testGetType(filename: c_string, dsetname: c_string) {
  extern proc c_getType(filename, colname, errMsg): c_int;
  extern proc c_free_string(a);
  extern proc strlen(a): int;
  var errMsg: c_ptr(uint(8));
  defer {
    c_free_string(errMsg);
  }
  var causeError = "asdasdasd":c_string;
  var arrowType = c_getType(filename, dsetname, c_ptrTo(errMsg));

  // a positive value corresponds to an arrow type
  // -1 corresponds to unsupported type
  if arrowType >= 0 {
    return 0;
  } else {
    var chplMsg;
    try! chplMsg = string.createCopyingBuffer(errMsg, strlen(errMsg));
    writeln(chplMsg);
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
  try! ret = string.createCopyingBuffer(cVersionString);
  try {
    // Ensure that version number can be cast to int
    // Not checking version number for compatability
    var vMajor = ret[0]:int;
    return 0;
  } catch {
    writeln("FAILED: c_getVersionInfo with ", ret);
    return 1;
  }
}

proc testGetDsets(filename) {
  extern proc c_getDatasetNames(f: c_string, r: c_ptr(c_ptr(c_char)), e: c_ptr(c_ptr(c_char))): int(32);
  extern proc c_free_string(ptr);
  extern proc strlen(a): int;
  var cDsetString: c_ptr(c_char);
  var errMsg: c_ptr(c_char);
  var st = c_getDatasetNames(filename, c_ptrTo(cDsetString), c_ptrTo(errMsg));
  defer {
    c_free_string(cDsetString);
    c_free_string(errMsg);
  }
  var ret;
  try! ret = string.createCopyingBuffer(cDsetString, strlen(cDsetString));

  if st == 0 && ret == "my-dset-name-test" {
    return 0;
  } else if st != 0 {
    var chplMsg;
    try! chplMsg = string.createCopyingBuffer(errMsg, strlen(errMsg));
    writeln(chplMsg);
    writeln("FAILED: c_getDatasetNames");
    return 1;
  } else {
    writeln("FAILED: c_getDatasetNames with ", ret);
    return 1;
  }
}

proc testReadStrings(filename, dsetname) {
  extern proc c_readColumnByName(filename, chpl_arr, colNum, numElems, startIdx, batchSize, errMsg): int;
  extern proc c_getStringColumnNumBytes(filename, colname, offsets, numElems, startIdx, errMsg): int;
  extern proc c_getNumRows(chpl_str, err): int;

  extern proc c_free_string(a);
  extern proc strlen(a): int;
  var errMsg: c_ptr(uint(8));
  defer {
    c_free_string(errMsg);
  }

  var size = c_getNumRows(filename, c_ptrTo(errMsg));
  var offsets: [0..#size] int;
  
  c_getStringColumnNumBytes(filename, dsetname, c_ptrTo(offsets[0]), size, 0, c_ptrTo(errMsg));
  var byteSize  = + reduce offsets;
  if byteSize < 0 {
    var chplMsg;
    try! chplMsg = string.createCopyingBuffer(errMsg, strlen(errMsg));
    writeln(chplMsg);
  }

  var a: [0..#byteSize] uint(8);

  if(c_readColumnByName(filename, c_ptrTo(a), dsetname, 3, 0, 1, c_ptrTo(errMsg)) < 0) {
    var chplMsg;
    try! chplMsg = string.createCopyingBuffer(errMsg, strlen(errMsg));
    writeln(chplMsg);
  }

  var localSlice = new lowLevelLocalizingSlice(a, 0..3);
  var firstElem = string.createAdoptingBuffer(localSlice.ptr, 3, 4);
  if firstElem == 'asd' {
    return 0;
  } else {
    writeln("FAILED: reading string file ", firstElem);
    return 1;
  }
  
  return 0;
}

proc testMultiDset() {
  const filename = 'resources/multi-col.parquet'.c_str();
  extern proc c_getDatasetNames(f: c_string, r: c_ptr(c_ptr(c_char)), e: c_ptr(c_ptr(c_char))): int(32);
  extern proc c_free_string(ptr);
  extern proc strlen(a): int;
  var cDsetString: c_ptr(c_char);
  var errMsg: c_ptr(c_char);
  var st = c_getDatasetNames(filename, c_ptrTo(cDsetString), c_ptrTo(errMsg));
  defer {
    c_free_string(cDsetString);
    c_free_string(errMsg);
  }
  var ret;
  try! ret = string.createCopyingBuffer(cDsetString, strlen(cDsetString));

  if st == 0 && ret == "col1,col2,col3" {
    return 0;
  } else if st != 0 {
    var chplMsg;
    try! chplMsg = string.createCopyingBuffer(errMsg, strlen(errMsg));
    writeln(chplMsg);
    writeln("FAILED: testMultiDset");
    return 1;
  } else {
    writeln("FAILED: testMultiDset with ", ret);
    return 1;
  }
}

proc main() {
  var errors = 0;

  const size = 1000;
  const filename = "myFile.parquet".c_str();
  const dsetname = "my-dset-name-test".c_str();

  const strFilename = "resources/strings.parquet".c_str();
  const strDsetname = "one".c_str();
  
  errors += testReadWrite(filename, dsetname, size);
  errors += testInt32Read();
  errors += testGetNumRows(filename, size);
  errors += testGetType(filename, dsetname);
  errors += testVersionInfo();
  errors += testGetDsets(filename);
  errors += testMultiDset();
  errors += testReadStrings(strFilename, strDsetname);

  if errors != 0 then
    writeln(errors, " Parquet tests failed");

  remove("myFile.parquet");
}
