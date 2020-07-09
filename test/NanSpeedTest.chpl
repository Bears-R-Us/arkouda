use TestBase;

use ReductionMsg;

config const ARRSIZE = 1_000_000;
config const GROUPS = 64;
config const MAX_VAL = 500_000;

proc testSkipNan(n:int, g:int) {
  var d: Diags;
  var orig = makeDistArray(n, int);
  fillInt(orig, 0, 500000);
  var segments = makeDistArray(g, int);
  fillInt(segments, 1, g+1);
  
  d.start();
  segMean(orig, segments, true);
  d.stop(printTime=false);

  return d.elapsed();
}

proc testRegular(n:int, g:int) {
  var d: Diags;
  var orig = makeDistArray(n, int);
  fillInt(orig, 0, 500000);
  var segments = makeDistArray(g, int);
  fillInt(segments, 1, g+1);
  
  d.start();
  segMean(orig, segments, false);
  d.stop(printTime=false);

  return d.elapsed();
}

proc main() {
  const elapsed = testSkipNan(ARRSIZE, GROUPS);
  const elapsedReg = testRegular(ARRSIZE,GROUPS);
  const MB = byteToMB(8*ARRSIZE):real;
  if printTimes {
    writeln("mean with SkipNan ran on %i elements (%.1dr MB) in %.2dr seconds (%.2dr MB/s)\n".format(ARRSIZE, MB, elapsed, MB/elapsed));
  writeln("mean without SkipNan ran on %i elements (%.1dr MB) in %.2dr seconds (%.2dr MB/s)\n".format(ARRSIZE, MB, elapsedReg, MB/elapsedReg));
  }
}
