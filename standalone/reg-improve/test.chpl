require "test.h";
require "test.o";

use CTypes;
use Time;
use BlockDist;

extern proc c_getNumRowGroups(readerIdx): c_int;
extern proc c_free(arr);
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
  var fillSizesT: stopwatch;
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
  var entrySeg = blockDist.createArray(0..#numElems, int);
  allocT.stop();
  var numCopied = 0;
  var byteSizes = 0;
  var vals: [0..1] c_ptr(MyByteArray);

  for i in 0..#numRowGroups {
    createReaders1T.start();
    c_createRowGroupReader(i, i);
    createReaders1T.stop();

    createReaders2T.start();
    c_createColumnReader(c_ptrTo(colname), i);
    createReaders2T.stop();
    
    var numRead = 0;
    readT.start();
    vals[i] = c_readParquetColumnChunks(c_ptrTo(filename), 8192, numElems, i, c_ptrTo(numRead)): c_ptr(MyByteArray);
    readT.stop();

    fillSizesT.start();
    forall (id, j) in zip(0..#numRead, numCopied..#numRead) with (+ reduce byteSizes) {
      ref curr = vals[i][id];
      entrySeg[j] = curr.len + 1;
      byteSizes += entrySeg[j];
    }
    fillSizesT.stop();
  }

  allocT.start();
  var entryVal = blockDist.createArray(0..#byteSizes, uint(8));
  allocT.stop();

  copyT.start();
  var idx = 0;
  for i in 0..#1000{
    var curr = vals[0][i];
    for j in 0..#curr.len {
      entryVal[idx] = curr.ptr[j];
      idx+=1;
    }
    idx+=1;
  }
  copyT.stop();
  c_free(vals[0]:c_ptr(void));
  c_free(vals[1]:c_ptr(void));
    
  writeln("Byte sizes: ", byteSizes);
  writeln(entrySeg[0..10]);
  writeln(entryVal[0..10]);
  t.stop();
  writeln("Read took                : ", readT.elapsed());
  writeln("Fill sizes took          : ", fillSizesT.elapsed());
  writeln("Copy took                : ", copyT.elapsed());
  writeln("Alloc took               : ", allocT.elapsed());
  writeln("Open files took          : ", openFilesT.elapsed());
  writeln("Create row readers took  : ", createReaders1T.elapsed());
  writeln("Create col readers took  : ", createReaders2T.elapsed());
  writeln("Total took               : ", t.elapsed());
}
