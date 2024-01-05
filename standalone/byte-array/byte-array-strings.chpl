/*
For the byte array optimization, we copy over the C++ buffers, rather than the
actual data, so we are using that to store our strings, requiring only a single
pass, removing the need for a segmnets array, and I'll probably forget to fill
in the rest later.
 */

use CTypes;
use Reflection;
use Time;
require "ArrowFunctions.h";
require "ArrowFunctions.o";

config const filename = '/Users/ben.mcdonald/arkouda/test-file.parquet';
config const colname = 'strings_array';
config const batchSize = 100;
extern var ARROWERROR: c_int;

extern record MyByteArray {
  var len: uint(32);
  var ptr: c_ptr(uint(8));
}

extern proc c_freeByteArray(input);

proc main() {
  var totalT: stopwatch;
  var t1: stopwatch;
  var t2: stopwatch;
  var t3: stopwatch;

  totalT.start();
  
  var filenames: [0..#1] string;
  filenames[0] = filename;

  t1.start();
  var len = getArrSize(filenames[0]);
  t1.stop();

  t2.start();
  var ret: [0..#len] bytes;
  t2.stop();
  
  t3.start();
  var pqList = readStrFilesByName(ret, filenames, colname);
  forall i in 0..#len {
    var curr = pqList[i];
    ret[i] = bytes.createBorrowingBuffer(curr.ptr, curr.len);
  }
  c_freeByteArray(pqList);
  t3.stop();

  totalT.stop();
  writeln("Total time     : ", totalT.elapsed());
  writeln("Get size       : ", t1.elapsed());
  writeln("Created seg    : ", t2.elapsed());
  writeln("Read files     : ", t3.elapsed());
}

proc getArrSize(filename: string) throws {
  extern proc c_getNumRows(chpl_str, errMsg): int;
  var pqErr = new parquetErrorMsg();

  var size = c_getNumRows(filename.localize().c_str(),
                          c_ptrTo(pqErr.errMsg));
  if size == ARROWERROR {
    pqErr.parquetError(getLineNumber(), getRoutineName(), getModuleName());
  }
  return size;
}

proc readStrFilesByName(ref A: [] ?t, filenames: [] string, dsetname: string) throws {
  extern proc c_readColumnByName(filename, chpl_arr, colNum, numElems, startIdx, batchSize, byteLength, errMsg): c_ptr(void);

  var pqErr = new parquetErrorMsg();

  var pqVals = c_readColumnByName(filename.localize().c_str(), c_ptrTo(A),
                        dsetname.localize().c_str(), A.size, 0,
                                  batchSize, -1, c_ptrTo(pqErr.errMsg));
  return pqVals: c_ptr(MyByteArray);
}

record parquetErrorMsg {
  var errMsg: c_ptr(uint(8));
  proc init() {
    errMsg = nil;
  }
    
  proc deinit() {
    extern proc c_free_string(ptr);
    c_free_string(errMsg);
  }

  proc parquetError(lineNumber, routineName, moduleName) throws {
    extern proc strlen(a): int;
    var err: string;
    try {
      err = string.createCopyingBuffer(errMsg, strlen(errMsg));
    } catch e {
      err = "Error converting Parquet error message to Chapel string";
    }
    throw new Error(err);
  }
}
