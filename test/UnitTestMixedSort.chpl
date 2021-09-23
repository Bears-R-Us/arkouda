private use MixedSort;
private use RadixSortLSD;
private use Random;
private use AryUtil;
private use TestBase;
private use CommAggregation;

private config const size = 100_000_000;
private config const pow:real = 1.5;

proc main() {
  const D = newBlockDom({0..#size});
  var y:[D] real;
  fillRandom(y);
  const lb = min reduce y;
  const ub = max reduce y;
  y = (y - lb) / (ub - lb);
  var x:[D] int = [yi in y] (2**32 * (yi ** (pow+1))): int;
  var d: Diags;

  d.start();
  var a1 = radixSortLSD_ranks(x);
  d.stop(printTime=false);
  if printTimes then writeln("radixSortLSD took %.2dr seconds (%.2dr MB/s)".format(d.elapsed(), byteToMB(size*8)/d.elapsed()));

  var x1:[D] int;
  forall (x1i, a1i) in zip(x1, a1) with (var agg = newSrcAggregator(int)) {
    agg.copy(x1i, x[a1i]);
  }
  if !isSorted(x1) {
    writeln("radixSortLSD failed to sort!");
  }
  
  d.start();
  var a2 = mixedSort_ranks(x);
  d.stop(printTime=false);
  if printTimes then writeln("mixedSort took %.2dr seconds (%.2dr MB/s)".format(d.elapsed(), byteToMB(size*8)/d.elapsed()));

  var x2:[D] int;
  forall (x2i, a2i) in zip(x2, a2) with (var agg = newSrcAggregator(int)) {
    agg.copy(x2i, x[a2i]);
  }
  if !isSorted(x2) {
    writeln("mixedSort failed to sort!");
  }
  writeln("Answers match? >>> ", && reduce (a1 == a2), " <<<");
}