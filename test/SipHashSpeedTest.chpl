use SipHash;
use RandArray;
use Time;
use BlockDist;

config const NINPUTS = 10_000;
config const INPUTSIZE = 8;
config const LENGTHS = "fixed";
config const COMMDEBUG = false;

proc testFixedLength(n:int, size:int) {
  var t = new Timer();
  const D1 = {0..#(n*size)}; 
  const DD1: domain(1) dmapped Block(boundingBox=D1) = D1;
  var buf: [DD1] uint(8);
  forall (b, i) in zip(buf, 0..) {
    b = i: uint(8);
  }
  const D2 = {0..#n};
  const DD2: domain(1) dmapped Block(boundingBox=D2) = D2;
  var hashes: [DD2] 2*uint(64);
  if COMMDEBUG { startCommDiagnostics(); }
  t.start();
  forall (h, i) in zip(hashes, DD2) {
    h = sipHash128(buf, i*size..#size);
  }
  t.stop();
  if COMMDEBUG { stopCommDiagnostics(); writeln(getCommDiagnostics());}
  return t.elapsed();
}

proc testVariableLength(n:int, meanSize:int) {
  var t = new Timer();
  const logMean:real = log(meanSize:real)/2;
  const logStd:real = sqrt(2*logMean);
  var (segs, vals) = newRandStringsLogNormalLength(n, logMean, logStd);
  const D = segs.domain;
  var lengths: [D] int;
  forall (l, s, i) in zip(lengths, segs, D) {
    if i == D.high then l = vals.size - s;
                   else l = segs[i+1] - s;
  }
  var hashes: [segs.domain] 2*uint(64);
  if COMMDEBUG { startCommDiagnostics(); }
  t.start();
  forall (h, i, l) in zip(hashes, segs, lengths) {
    h = sipHash128(vals, i..#l);
  }
  t.stop();
  if COMMDEBUG { stopCommDiagnostics(); writeln(getCommDiagnostics());}
  return (t.elapsed(), vals.size);
}

proc main() {
  if LENGTHS == "fixed" {
    var elapsed = testFixedLength(NINPUTS, INPUTSIZE);
    writeln("Hashed %i blocks (%i bytes) in %.2dr seconds\nRate = %.2dr MB/s".format(NINPUTS, NINPUTS*INPUTSIZE, elapsed, (NINPUTS*INPUTSIZE)/(1024*1024*elapsed)));
  } else if LENGTHS == "variable" {
    var (elapsed, nbytes) = testVariableLength(NINPUTS, INPUTSIZE);
    writeln("Hashed %i blocks (%i bytes) in %.2dr seconds\nRate = %.2dr MB/s".format(NINPUTS, nbytes, elapsed, (nbytes)/(1024*1024*elapsed)));
  }    
}
