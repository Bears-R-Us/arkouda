use TestBase;

use SipHash;

config const NINPUTS = 100_000;
config const INPUTSIZE = 64;
config const SEED = "none";

config const computeOnSegments = false;

enum testMode {fixed, variable};
config const mode = testMode.variable;
config const compareTypes = false;

proc testFixedLength(n:int, size:int, type t) {
  var d: Diags;
  var buf = makeDistArray(n*size, t);
  forall (b, i) in zip(buf, 0..) {
    b = i: t;
  }
  var hashes = makeDistArray(n, 2*uint(64));
  if computeOnSegments {
    use SegmentedComputation;
    var segs = makeDistArray(n, int);
    forall (i, s) in zip(segs.domain, segs) {
      s = i*size;
    }
    d.start();
    hashes = computeOnSegments(segs, buf, SegFunction.SipHash128, 2*uint(64));
    d.stop(printTime=false);
  } else {
    d.start();
    forall (h, i) in zip(hashes, hashes.domain) {
      h = sipHash128(buf, i*size..#size);
    }
    d.stop(printTime=false);
  }
  return (d.elapsed(), n*size*numBytes(t));
}

proc testVariableLength(n:int, meanSize:int, type t) {
  var d: Diags;
  const logMean:real = log(meanSize:real)/2;
  const logStd:real = sqrt(2*logMean);
  var (segs, vals) = newRandStringsLogNormalLength(n, logMean, logStd, seedStr=SEED);
  var tohash: [vals.domain] t = [v in vals] v: t;
  const D = segs.domain;
  var lengths: [D] int;
  forall (l, s, i) in zip(lengths, segs, D) {
    if i == D.high then l = vals.size - s;
                   else l = segs[i+1] - s;
  }
  var hashes: [segs.domain] 2*uint;
  d.start();
  if computeOnSegments {
    use SegmentedComputation;
    hashes = computeOnSegments(segs, vals, SegFunction.SipHash128, 2*uint(64));
  } else {
    forall (h, i, l) in zip(hashes, segs, lengths) {
      h = sipHash128(tohash, i..#l);
    }
  }
  d.stop(printTime=false);
  return (d.elapsed(), vals.size * numBytes(t));
}

proc main() {
  var (elapsed, nbytes) = if mode == testMode.variable then testVariableLength(NINPUTS, INPUTSIZE, uint(8))
    else testFixedLength(NINPUTS, INPUTSIZE, uint(8));
  var MB = byteToMB(nbytes);
  if printTimes then
    writeln("Hashed %i uint(8) blocks (%.1dr MB) in %.2dr seconds (%.2dr MB/s)\n".format(NINPUTS, MB, elapsed, MB/elapsed));

  if compareTypes {
    (elapsed, nbytes) = if mode == testMode.variable then testVariableLength(NINPUTS, INPUTSIZE, int(64))
      else testFixedLength(NINPUTS, INPUTSIZE, int(64));
    MB = byteToMB(nbytes);
    if printTimes then
      writeln("Hashed %i int(64) blocks (%.1dr MB) in %.2dr seconds (%.2dr MB/s)\n".format(NINPUTS, MB, elapsed, MB/elapsed));

    (elapsed, nbytes) = if mode == testMode.variable then testVariableLength(NINPUTS, INPUTSIZE, real(64))
      else testFixedLength(NINPUTS, INPUTSIZE, real(64));
    MB = byteToMB(nbytes);
    if printTimes then
      writeln("Hashed %i real(64) blocks (%.1dr MB) in %.2dr seconds (%.2dr MB/s)\n".format(NINPUTS, MB, elapsed, MB/elapsed));
  }
}
