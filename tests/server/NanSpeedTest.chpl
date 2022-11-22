use TestBase;

use ReductionMsg;

config const ARRSIZE = 10000;
config const GROUPS = 64;
config const MAX_VAL = 500_000;

proc testSkipNan(orig, segments) {
  var d: Diags;
  
  d.start();
  var res = segMin(orig, segments, true);
  d.stop(printTime=false);

  return (d.elapsed(), res);
}

proc testRegular(orig, segments) {
  var d: Diags;
  
  d.start();
  var res = segMin(orig, segments, false);
  d.stop(printTime=false);

  return (d.elapsed(), res);
}

proc main() {
  var orig = makeDistArray(ARRSIZE, real);
  fillReal(orig, 0.0, 500000.0);
  var segments = makeDistArray(GROUPS, int);
  fillInt(segments, 1, GROUPS+1);
  
  const (elapsed, skipRes) = testSkipNan(orig,segments);
  const (elapsedReg, regRes) = testRegular(orig,segments);

  assert(skipRes.equals(regRes));
  
  const MB = byteToMB(8*ARRSIZE):real;
  if printTimes {
    writeln("mean with SkipNan ran on %i elements (%.1dr MB) in %.2dr seconds (%.2dr MB/s)\n".format(ARRSIZE, MB, elapsed, MB/elapsed));
  writeln("mean without SkipNan ran on %i elements (%.1dr MB) in %.2dr seconds (%.2dr MB/s)\n".format(ARRSIZE, MB, elapsedReg, MB/elapsedReg));
  }
}
