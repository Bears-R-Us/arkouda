use TestBase;

use ArraySetops;

config const NINPUTS = 10_000;
config const MAX_VAL = 50_000;

proc testIntersect1d(n:int) {
  var d: Diags;
  var a = makeDistArray(n, int);
  var b = makeDistArray(n, int);
  fillInt(a, 0, MAX_VAL);
  fillInt(b, 0, MAX_VAL);

  d.start();
  var in1d = intersect1d(a,b,false);
  d.stop(printTime=false);
  return d.elapsed();
}

proc testSetDiff1d(n:int) {
  var d: Diags;
  var a = makeDistArray(n, int);
  var b = makeDistArray(n, int);
  fillInt(a, 0, MAX_VAL);
  fillInt(b, 0, MAX_VAL);

  d.start();
  setdiff1d(a,b,false);
  d.stop(printTime=false);
  return d.elapsed();
}

proc testSetXor1d(n:int) {
  var d: Diags;
  var a = makeDistArray(n, int);
  var b = makeDistArray(n, int);
  fillInt(a, 0, MAX_VAL);
  fillInt(b, 0, MAX_VAL);

  d.start();
  setxor1d(a,b,false);
  d.stop(printTime=false);
  return d.elapsed();
}
proc testUnion1d(n:int) {
  var d: Diags;
  var a = makeDistArray(n, int);
  var b = makeDistArray(n, int);
  fillInt(a, 0, MAX_VAL);
  fillInt(b, 0, MAX_VAL);

  d.start();
  union1d(a,b);
  d.stop(printTime=false);
  return d.elapsed();
}

proc main() {
  var a = makeDistArray(NINPUTS, int);
  var b = makeDistArray(NINPUTS, int);
  fillInt(a, 0, MAX_VAL);
  fillInt(b, 0, MAX_VAL);

  const elapsedIntersect = testIntersect1d(NINPUTS);
  const elapsedDiff = testSetDiff1d(NINPUTS);
  const elapsedXor = testSetXor1d(NINPUTS);
  const elapsedUnion = testUnion1d(NINPUTS);

  const MB:real = byteToMB(NINPUTS*8.0);
  if printTimes {
    writeln("intersect1d on %i elements (%.1dr MB) in %.2dr seconds (%.2dr MB/s)".format(NINPUTS, MB, elapsedIntersect, MB/elapsedIntersect));
    writeln("setdiff1d on %i elements (%.1dr MB) in %.2dr seconds (%.2dr MB/s)".format(NINPUTS, MB, elapsedDiff, MB/elapsedDiff));
    writeln("setxor1d on %i elements (%.1dr MB) in %.2dr seconds (%.2dr MB/s)".format(NINPUTS, MB, elapsedXor, MB/elapsedXor));
    writeln("union1d on %i elements (%.1dr MB) in %.2dr seconds (%.2dr MB/s)".format(NINPUTS, MB, elapsedUnion, MB/elapsedUnion));
  }
}