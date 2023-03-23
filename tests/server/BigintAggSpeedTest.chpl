use TestBase;
use Random;
use CommAggregation;
use BigInteger;

const numTasks = numLocales * here.maxTaskPar;
config const N = 1000; // number of updates per task
config const M = 100;   // number of entries in the table per task

const numUpdates = N * numTasks;
const tableSize = M * numTasks;

proc testit(type elemType) {
  writeln(elemType:string);
  const D = makeDistDom(tableSize);
  var A: [D] elemType;
  forall (a, d) in zip(A, D) do a = d;

  const UpdatesDom = makeDistDom(numUpdates);
  var Rindex: [UpdatesDom] int;

  fillRandom(Rindex, 208);
  Rindex = mod(Rindex, tableSize);
  var tmp: [UpdatesDom] elemType;
  forall t in tmp do t = -1;

  var d: Diags;

  d.start();
  forall (t, r) in zip (tmp, Rindex) with (var agg = newSrcAggregator(elemType)) {
    agg.copy(t, A[r]);
  }
  d.stop("Gather ");

  d.start();
  forall (t, r) in zip (tmp, Rindex) with (var agg = newDstAggregator(elemType)) {
    agg.copy(A[r], t);
  }
  d.stop("Scatter");
  writeln();
}

proc main() {
  testit(int);
  testit(bigint);
}
