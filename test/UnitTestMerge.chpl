use Merge;
use RadixSortLSD;
use Random;

proc testBinarySearch(size, trials) {
  var a = makeDistArray(size, int);
  fillRandom(a);
  var b = radixSortLSD_keys(a);
  var R = new RandomStream(int);
  for t in 0..#trials {
    var x = R.getNext();
    var hit = binarySearch(b, x);
    var (cond, loc) = maxloc reduce zip((b >= x), b.domain);
    if cond && (hit != loc) {
      writeln("Binary search failed on trial ", t); stdout.flush();
      writeln("x = %t, res = %t, b[res] = %t, ans = %t, b[ans] = %t".format(x, hit, b[hit], loc, b[loc])); stdout.flush();
      return false;
    }
  }
  writeln("Passed binary search test");
  return true;
}

proc testMerge(n, m, minlen, maxlen) {
  var a = makeDistArray(n, int);
  fillInt(a, nmin, nmax);
  var sa = radixSortLSD_keys(a);
  var b = makeDistArray(m, int);
  fillInt(b, mmin, mmax);
  var sb = radixSortLSD_keys(b);
  
}

config const N = 1_000;
config const TRIALS = 10;
config const M = 100;
config const MINLEN = 1;
config const MAXLEN = 10;

proc main() {
  if testBinarySearch(N, TRIALS) {
    testMerge(N, M, NMIN, NMAX, MMIN, MMAX);
  }
}