use ObjectPool;
use LisExprData;
use Time;

config const size = 100000;

proc main() {
  for i in 0..3 do
    testPoolPerf();
  for i in 0..3 do
    testAllocPerf();
}

proc testAllocPerf() {
  var t: Timer;
  t.start();
  for i in 0..#size {
    var a = new unmanaged ValueClass(i);
    a.v += 1;
    delete a;
  }
  t.stop();
  writeln("Alloc took            : ", t.elapsed());
}

proc testPoolPerf() {
  var t: Timer;
  t.start();
  var p = new pool();
  for i in 0..#size {
    var a = p.getInt(i);
    a.v = i;
    a.v += 1;
    p.freeInt(a);
  }
  t.stop();
  writeln("Pool alloc took      : ", t.elapsed());
}
