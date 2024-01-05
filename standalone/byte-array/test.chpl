require "test.h";
require "test.o";

use CTypes;

extern record MyByteArray {
  var len: uint(32);
  var ptr: c_ptr(uint(8));
};

proc main() {
  extern proc c_getByteArray(): c_ptr(void);
  extern proc c_freeByteArray(a);
  var asd = c_getByteArray(): c_ptr(MyByteArray);
  for i in 0..#10 {
    var curr = asd[i];
    writeln(curr);
    writeln(bytes.createBorrowingBuffer(curr.ptr, curr.len));
  }
  c_freeByteArray(asd);
}
