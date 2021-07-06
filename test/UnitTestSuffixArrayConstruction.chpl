use TestBase;
use SuffixArrayConstruction;


proc testSuffixArraySkew(byteString: bytes, expectedSuffixArray: [] int) throws {
  var byteArray = byteString.bytes();
  var d: Diags;
  var size: int = byteArray.size;

  // Follows segSuffixArrayMsg in SegmentedMsg.chpl including padding with 0s 
  var integerArray: [0..size+2] int;
  var tmpArray: [0..size+2] int;
  var suffixArray: [0..size-1] int;

  var x:int;
  var y:int;
  forall (x, y) in zip(integerArray[0..size-1], byteArray[0..size-1]) do x=y;
  integerArray[size]=0;
  integerArray[size+1]=0;
  integerArray[size+2]=0;

  writeln(">>> SuffixArrayConstruction for %s".format(byteString.decode().strip('\x00'))); stdout.flush();
  d.start();
  SuffixArraySkew(integerArray, tmpArray, size, 256);
  d.stop("SuffixArrayConstruction");
  for (x, y) in zip(suffixArray[0..size-1], tmpArray[0..size-1])  do x = y;
  writeln("Expected:\n%t".format(expectedSuffixArray));
  writeln("Actual:\n%t".format(suffixArray));
}

proc main() {
  // "yabbadabbado" follows the example in Section 3 from "Linear Work Suffix Array Construction"
  try! testSuffixArraySkew(b"yabbadabbado\x00", [12, 1, 6, 4, 9, 3, 8, 2, 7, 5, 10, 11, 0]);

  // "banana" follows the example in the docstring of suffix_array on the client-side
  try! testSuffixArraySkew(b"banana\x00", [6, 5, 3, 1, 0, 4, 2]);
}
