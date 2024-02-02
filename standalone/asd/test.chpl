require "test.h";
require "test.o";

use CTypes;

extern record MyByteArray {
  var len: uint(32);
  var ptr: c_ptr(uint(8));
};

proc main() {
  extern proc c_getParquetReader(a, b, c);
  extern proc c_readParquetColumn(reader, numElems, batchSize): c_ptr(void);
  extern proc c_readParquetColumnChunks(filename, colname, batchSize, numElems): c_ptr(void);

  var reader: c_ptr(void) = "":c_ptr(void);
  var filename = "test-file_LOCALE0000";
  var colname = "strings_array";
  c_getParquetReader(reader, c_ptrTo(filename), c_ptrTo(colname));

  var vals = c_readParquetColumnChunks(c_ptrTo(filename), c_ptrTo(colname), 8192, (8192*12)): c_ptr(MyByteArray);
  for i in 0..#100 {
    var curr = vals[i];
    writeln(bytes.createBorrowingBuffer(curr.ptr, curr.len));
  }

  /*
  var asd = c_getByteArray(): c_ptr(MyByteArray);
  for i in 0..#10 {
    var curr = asd[i];
    writeln(curr);
    writeln(bytes.createBorrowingBuffer(curr.ptr, curr.len));
  }
  c_freeByteArray(asd);
  */
}
