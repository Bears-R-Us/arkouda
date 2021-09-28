private use MixedSort;
private use RadixSortLSD;
private use Random;
private use AryUtil;
private use TestBase;
private use CommAggregation;

config const size = 10_000;
config const pow:real = 1.5;
config const toy: bool = false;

proc doToy() {
  const toysize = if (size <= 1000) then size else 50;
  const D = newBlockDom({0..#toysize});
  var x:[D] int;
  forall i in D {
    x[i] = toysize * (toysize - i - 1);
  }
  writeln("Sorting...");
  var d: Diags;
  d.start();
  var a = mixedSort_ranks(x);
  d.stop(printTime=true);
  writeln("Permuting...");
  var xs:[D] int;
  forall (xsi, ai) in zip(xs, a) with (var agg = newSrcAggregator(int)) {
    agg.copy(xsi, x[ai]);
  }
  var success = isSorted(xs);
  if !success {
    writeln("Failed to sort!");
    writeln("Idx Original Sorted Argsort");
    for i in D {
      writeln("%3i %04xu %04xu %3i".format(i, x[i], xs[i], a[i]));
    }
  } else {
    writeln("Success!");
  }
  return success:int;
}

proc main() {
  if toy {
    return doToy();
  }
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
    if size <= 100 {
      writeln("idx Original   Sorted   Attempted   Sorted idx  Attempted idx");
      for i in D {
        writeln("%3i %016xu %016xu %016xu %3i %3i".format(i, x[i], x1[i], x2[i], a1[i], a2[i]));
      }
    }
  }
  var success = && reduce (a1 == a2);
  writeln("Answers match? >>> ", success, " <<<");
  return success:int;
}
