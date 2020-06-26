use TestBase;

use SipHash;

config const NINPUTS = 100_000;
config const INPUTSIZE = 64;

enum testMode {fixed, variable};
config const mode = testMode.variable;

proc testFixedLength(n:int, size:int) {
  var d: Diags;
  var buf = makeDistArray(n*size, uint(8));
  forall (b, i) in zip(buf, 0..) {
    b = i: uint(8);
  }
  var hashes = makeDistArray(n, 2*uint(64));
  d.start();
  forall (h, i) in zip(hashes, hashes.domain) {
    h = sipHash128(buf, i*size..#size);
  }
  d.stop(printTime=false);
  return (d.elapsed(), n*size);
}

proc testVariableLength(n:int, meanSize:int) {
  var d: Diags;
  const logMean:real = log(meanSize:real)/2;
  const logStd:real = sqrt(2*logMean);
  var (segs, vals) = newRandStringsLogNormalLength(n, logMean, logStd);
  const D = segs.domain;
  var lengths: [D] int;
  forall (l, s, i) in zip(lengths, segs, D) {
    if i == D.high then l = vals.size - s;
                   else l = segs[i+1] - s;
  }
  var hashes: [segs.domain] 2*uint;
  d.start();
  forall (h, i, l) in zip(hashes, segs, lengths) {
    h = sipHash128(vals, i..#l);
  }
  d.stop(printTime=false);
  return (d.elapsed(), vals.size);
}

proc main() {
  const (elapsed, nbytes) = if mode == testMode.variable then testVariableLength(NINPUTS, INPUTSIZE)
                                                         else testFixedLength(NINPUTS, INPUTSIZE);
  const MB = byteToMB(nbytes);
  if printTimes then
    writeln("Hashed %i blocks (%.1dr MB) in %.2dr seconds (%.2dr MB/s)\n".format(NINPUTS, MB, elapsed, MB/elapsed));
}
