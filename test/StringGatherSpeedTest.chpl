use SegmentedArray;
use RandArray;
use MultiTypeSymbolTable;
use RadixSortLSD;

config const N = 10_000;
config const MEANLEN = 10;

proc testPermute(n:int, meanLen: numeric) {
  var st = new owned SymTab();
  var t = new Timer();
  const logMean = log(meanLen:real)/2;
  const logStd = sqrt(2*logMean);
  var (segs, vals) = newRandStringsLogNormalLength(n, logMean, logStd);
  var strings = new owned SegString(segs, vals, st);
  var rint: [segs.domain] int;
  fillInt(rint, 0, max(int));
  var perm = radixSortLSD_ranks(rint);
  t.start();
  var (psegs, pvals) = strings[perm];
  t.stop();
  return (t.elapsed(), vals.size);
}

proc main() {
  var (elapsed, size) = testPermute(N, MEANLEN);
  writeln("Permuted %i strings (%i bytes) in %t seconds".format(N, size, elapsed));
  writeln("Rate = %t MB/s".format(size / (1024 * 1024 * elapsed)));
}