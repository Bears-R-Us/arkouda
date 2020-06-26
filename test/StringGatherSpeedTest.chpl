use TestBase;

use RadixSortLSD;

config const N = 100_000;
config const MEANLEN = 10;

proc testPermute(n:int, meanLen: numeric) {
  var st = new owned SymTab();
  var d: Diags;
  const logMean = log(meanLen:real)/2;
  const logStd = sqrt(2*logMean);
  var (segs, vals) = newRandStringsLogNormalLength(n, logMean, logStd);
  var strings = new owned SegString(segs, vals, st);
  var rint: [segs.domain] int;
  fillInt(rint, 0, max(int));
  var perm = radixSortLSD_ranks(rint);
  d.start();
  var (psegs, pvals) = strings[perm];
  d.stop(printTime=false);
  return (d.elapsed(), vals.size);
}

proc main() {
  var (elapsed, nbytes) = testPermute(N, MEANLEN);
  const MB = byteToMB(nbytes);
  if printTimes then
    writef("Permuted %i strings (%.1dr MB) in %.2dr seconds (%.2dr MB/s)\n", N, MB, elapsed, MB/elapsed);
}
