use SipHash;
use Time;
use BlockDist;

config const NINPUTS = 10_000;
config const INPUTSIZE = 8;
config param COMMDEBUG = false;

proc main() {
  const D1 = {0..#(NINPUTS*INPUTSIZE)}; 
  const DD1: domain(1) dmapped Block(boundingBox=D1) = D1;
  var buf: [DD1] uint(8);
  forall (b, i) in zip(buf, 0..) {
    b = i: uint(8);
  }
  const D2 = {0..#NINPUTS};
  const DD2: domain(1) dmapped Block(boundingBox=D2) = D2;
  var hashes: [DD2] 2*uint(64);
  if COMMDEBUG { startCommDiagnostics(); }
  var t = getCurrentTime();
  forall (h, i) in zip(hashes, DD2) {
    h = sipHash128(buf, i*INPUTSIZE..#INPUTSIZE);
  }
  var elapsed = getCurrentTime() - t;
  if COMMDEBUG { stopCommDiagnostics(); writeln(getCommDiagnostics());}
  writeln("Hashed %i blocks (%i bytes) in %.2dr seconds\nRate = %.2dr MB/s".format(NINPUTS, NINPUTS*INPUTSIZE, elapsed, (NINPUTS*INPUTSIZE)/(1024*1024*elapsed)));
}
