require "test.h";
require "test.o";

use CTypes;
use Time;
use BlockDist;

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
  var t: stopwatch;
  t.start();
  var readT: stopwatch;
  var copyT: stopwatch;

  var createReaders1T: stopwatch;
  var createReaders2T: stopwatch;
  var openFilesT: stopwatch;
  
  var filename = "test-file_LOCALE0000";
  var colname = "strings_array";
  var numElems = 100000000;

  openFilesT.start();
  c_openFile(c_ptrTo(filename), 0);
  c_openFile(c_ptrTo(filename), 1);
  openFilesT.stop();
  var numRowGroups = c_getNumRowGroups(0);
  var allocT: stopwatch;
  allocT.start();
  var ret = blockDist.createArray(0..#numElems, bytes);
  allocT.stop();
  var numCopied = 0;

  for i in 0..#numRowGroups {
    createReaders1T.start();
    c_createRowGroupReader(i, i);
    createReaders1T.stop();

    createReaders2T.start();
    c_createColumnReader(c_ptrTo(colname), i);
    createReaders2T.stop();
    
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
  t.stop();
  writeln("Read took                : ", readT.elapsed());
  writeln("Copy took                : ", copyT.elapsed());
  writeln("Alloc took               : ", allocT.elapsed());
  writeln("Open files took          : ", openFilesT.elapsed());
  writeln("Create row readers took  : ", createReaders1T.elapsed());
  writeln("Create col readers took  : ", createReaders2T.elapsed());
  writeln("Total took               : ", t.elapsed());
}
