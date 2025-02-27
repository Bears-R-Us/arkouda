use ParquetMsg, CTypes, FileSystem;
use UnitTest;
use TestBase;

private config const ROWGROUPS = 512*1024*1024 / numBytes(int); // 512 mb of int64

type c_string = c_ptrConst(c_char);

extern proc c_readColumnByName(filename, arr_chpl, where_null_chpl, colNum,
                               numElems, startIdx, batchSize, byteLength,
                               hasNonFloatNulls, errMsg): int;
extern proc c_writeColumnToParquet(filename, arr_chpl, colnum,
                                   dsetname, numelems, rowGroupSize,
                                   dtype, compression, errMsg): int;
extern proc c_getStringColumnNumBytes(filename, colname, offsets, numElems,
                                      startIdx, batchSize, errMsg): int;
extern proc c_getDatasetNames(f: c_string, r: c_ptr(c_ptr(c_char)),
                              readNested, e: c_ptr(c_ptr(c_char))): int(32);
extern proc c_writeMultiColToParquet(filename: c_string,
                                     column_names: c_ptr(void),
                                     ptr_arr: c_ptr(c_ptr(void)),
                                     offset_arr: c_ptr(c_ptr(void)),
                                     objTypes: c_ptr(void),
                                     datatypes: c_ptr(void),
                                     segArr_sizes: c_ptr(void),
                                     colnum: int,
                                     numelems: int,
                                     rowGroupSize: int,
                                     compression: int,
                                     errMsg: c_ptr(c_ptr(c_uchar))): int;

extern proc c_getNumRows(chpl_str, err): int;
extern proc c_getType(filename, colname, errMsg): c_int;
extern proc c_getVersionInfo(): c_ptrConst(c_char);

extern proc c_free_string(a);
extern proc strlen(a): int;

proc testReadWrite(filename: c_string, dsetname: c_string, size: int) {
  extern proc c_free_string(a);
  extern proc strlen(a): int;
  var errMsg: c_ptr(uint(8));
  defer {
    c_free_string(errMsg);
  }
  var causeError = "cause-error":c_string;
  
  var a: [0..#size] int;
  for i in 0..#size do a[i] = i;

  if c_writeColumnToParquet(filename, c_ptrTo(a), 0, dsetname, size, 10000,
      ARROWINT64, 1, errMsg) < 0 {
    var chplMsg;
    try! chplMsg = string.createCopyingBuffer(errMsg, strlen(errMsg));
    writeln(chplMsg);
  }

  var b: [0..#size] int;

  if(c_readColumnByName(filename, c_ptrTo(b), false, dsetname, size, 0, 10000,
                        -1, false, c_ptrTo(errMsg)) < 0) {
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
  var errMsg: c_ptr(uint(8));
  defer {
    c_free_string(errMsg);
  }
  
  var a: [0..#50] int;
  var expected: [0..#50] int;
  for i in 0..#50 do expected[i] = i;
  
  if(c_readColumnByName("resources/int32.parquet".c_str(), c_ptrTo(a), false,
                        "array".c_str(), 50, 0, 1, -1, false, c_ptrTo(errMsg)) < 0) {
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
  var cDsetString: c_ptr(c_char);
  var errMsg: c_ptr(c_char);
  var st = c_getDatasetNames(filename, c_ptrTo(cDsetString), false,
                             c_ptrTo(errMsg));
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
  var errMsg: c_ptr(uint(8));
  defer {
    c_free_string(errMsg);
  }

  var size = c_getNumRows(filename, c_ptrTo(errMsg));
  var offsets: [0..#size] int;
  
  c_getStringColumnNumBytes(filename, dsetname, c_ptrTo(offsets[0]), size, 0,
                            256, c_ptrTo(errMsg));
  var byteSize  = + reduce offsets;
  if byteSize < 0 {
    var chplMsg;
    try! chplMsg = string.createCopyingBuffer(errMsg, strlen(errMsg));
    writeln(chplMsg);
  }

  var a: [0..#byteSize] uint(8);

  if(c_readColumnByName(filename, c_ptrTo(a), false, dsetname, 3, 0, 1, -1,
                        false, c_ptrTo(errMsg)) < 0) {
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
  var cDsetString: c_ptr(c_char);
  var errMsg: c_ptr(c_char);
  var st = c_getDatasetNames(filename, c_ptrTo(cDsetString), false, c_ptrTo(errMsg));
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

proc testMultiColWriteIntInt() {

  type elemType = int;

  const numCols = 4;
  const numElems = 10;

  var colNames = [i in 0..#numCols] ("col"+i:string).buff;
  var Arrs: [0..#numCols][0..#numElems] elemType;

  for col in Arrs.domain {
    for row in Arrs[col].domain {
      Arrs[col][row] = col+row;
    }
  }

  var ArrPtrs: [0..#numCols] c_ptr(elemType);

  for (ptr, Arr) in zip(ArrPtrs, Arrs) {
    ptr = c_ptrTo(Arr);
  }

  var ObjTypes = [0..#numCols] 1; // 1 is PDARRAY
  var DataTypes = [0..#numCols] ARROWINT64;

  var errStr = "E"*200;

  c_writeMultiColToParquet(filename="testMultiColWrite.parquet":c_string,
                           column_names=c_ptrTo(colNames),
                           ptr_arr=c_ptrTo(ArrPtrs):c_ptr(c_ptr(void)),
                           offset_arr=nil,
                           objTypes=c_ptrTo(ObjTypes),
                           datatypes=c_ptrTo(DataTypes),
                           segArr_sizes=nil,
                           colnum=numCols,
                           numelems=numElems,
                           rowGroupSize=ROWGROUPS,
                           compression=0,
                           errMsg=c_ptrTo(errStr.buff));

  return 0;
}

proc testMultiColWriteIntBool() {

  const numCols = 2;
  const numElems = 10;

  proc createArray(type elemType) {
    var Arr: [0..#numElems] elemType;
    return Arr;
  }

  var colNames = [i in 0..#numCols] ("col"+i:string).buff;

  var ArrInt = createArray(int);
  var ArrBool = createArray(bool);

  var ArrPtrs: [0..#numCols] c_ptr(void);

  ArrPtrs[0] = c_ptrTo(ArrInt);
  ArrPtrs[1] = c_ptrTo(ArrBool);

  var ObjTypes = [0..#numCols] 1; // 1 is PDARRAY

  var DataTypes = [0..#numCols] ARROWINT64;

  DataTypes[0] = ARROWINT64;
  DataTypes[1] = ARROWBOOLEAN;

  var errStr = "E"*200;

  c_writeMultiColToParquet(filename="testMultiColWrite.parquet":c_string,
                           column_names=c_ptrTo(colNames),
                           ptr_arr=c_ptrTo(ArrPtrs):c_ptr(c_ptr(void)),
                           offset_arr=nil,
                           objTypes=c_ptrTo(ObjTypes),
                           datatypes=c_ptrTo(DataTypes),
                           segArr_sizes=nil,
                           colnum=numCols,
                           numelems=numElems,
                           rowGroupSize=ROWGROUPS,
                           compression=0,
                           errMsg=c_ptrTo(errStr.buff));

  return 0;
}

proc testMultiColWriteIntSegArr() {

  record fakeSegArray {
    var totalSize: int;
    var SegmentsDom = {1..0};
    var Segments: [SegmentsDom] int;
    var Data: [0..<totalSize] int;
    var Sizes: [Segments.domain] int;

    proc init(totalSize, Segments) {
      this.totalSize = totalSize;
      this.SegmentsDom = Segments.domain;
      this.Segments = Segments;

      init this;

      for i in Segments.domain {
        if i == Segments.domain.high then
          Sizes[i] = totalSize - Segments[i];
        else
          Sizes[i] = Segments[i+1] - Segments[i];
      }
    }
  }

  const numCols = 2;
  const numElems = 4;

  proc createArray(type elemType) {
    var Arr: [0..#numElems] elemType;
    return Arr;
  }

  var colNames = [i in 0..#numCols] ("col"+i:string).buff;

  var ArrInt = createArray(int);
  var SegArr = new fakeSegArray(20, [0, 2, 3, 6]);

  var ArrPtrs: [0..#numCols] c_ptr(void);

  ArrPtrs[0] = c_ptrTo(ArrInt);
  ArrPtrs[1] = c_ptrTo(SegArr.Data);

  var OffsetArr: [0..#numCols] c_ptr(void);
  OffsetArr[1] = c_ptrTo(SegArr.Segments);

  var ObjTypes = [ObjType.PDARRAY, ObjType.SEGARRAY];
  var DataTypes = [0..#numCols] ARROWINT64;

  var SegArrSizes: [0..#numCols] c_ptr(void);
  SegArrSizes[1] = c_ptrTo(SegArr.Sizes);

  var errStr = "E"*200;

  c_writeMultiColToParquet(filename="testMultiColWrite.parquet":c_string,
                           column_names=c_ptrTo(colNames),
                           ptr_arr=c_ptrTo(ArrPtrs):c_ptr(c_ptr(void)),
                           offset_arr=c_ptrTo(OffsetArr),
                           objTypes=c_ptrTo(ObjTypes),
                           datatypes=c_ptrTo(DataTypes),
                           segArr_sizes=c_ptrTo(SegArrSizes),
                           colnum=numCols,
                           numelems=numElems,
                           rowGroupSize=ROWGROUPS,
                           compression=0,
                           errMsg=c_ptrTo(errStr.buff));

  return 0;
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
  errors += testMultiColWriteIntInt();
  errors += testMultiColWriteIntBool();
  errors += testMultiColWriteIntSegArr();

  if errors != 0 then
    writeln(errors, " Parquet tests failed");

  remove("myFile.parquet");
}
