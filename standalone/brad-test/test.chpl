require "test.h";
require "test.o";

use CTypes;
use Time;

extern proc c_getNumRowGroups(readerIdx): c_int;
extern proc c_openFile(filename, idx);
extern proc c_createRowGroupReader(rowGroup, readerIdx);
extern proc c_createColumnReader(colname, readerIdx);
extern proc c_readParquetColumnChunks(filename, batchSize,
                                      numElems, readerIdx, numRead): c_ptr(void);

extern record MyByteArray {
  var len: uint(32);
  var ptr: c_ptr(uint(8));
};

proc main() {
  var readT: stopwatch;
  var copyT: stopwatch;
  var filename = "test-file_LOCALE0000";
  var colname = "strings_array";
  var numElems = 100000000;

  c_openFile(c_ptrTo(filename), 0);
  c_openFile(c_ptrTo(filename), 1);
  var numRowGroups = c_getNumRowGroups(0);
  var ret: [0..#numElems] bytes;
  var numCopied = 0;

  for i in 0..#numRowGroups {
    c_createRowGroupReader(i, i);
    c_createColumnReader(c_ptrTo(colname), i);
    var numRead = 0;
    readT.start();
    var vals = c_readParquetColumnChunks(c_ptrTo(filename), 8192, numElems, i, c_ptrTo(numRead)): c_ptr(MyByteArray);
    readT.stop();

    copyT.start();
    forall (id, j) in zip(0..#numRead, numCopied..#numRead) {
      var curr = vals[id];
      ret[j] = bytes.createBorrowingBuffer(curr.ptr, curr.len);
    }
    copyT.stop();
    numCopied += numRead;
  }
  writeln("Read took       : ", readT.elapsed());
  writeln("Copy took       : ", copyT.elapsed());
}
