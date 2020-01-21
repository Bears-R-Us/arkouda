use Merge;
use RadixSortLSD;
use Random;
use RandArray;

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

proc testMerge(n, m, minLen, maxLen) {
  var st = new owned SymTab();
  var t = new Timer();
  writeln("Generating random strings..."); stdout.flush();
  t.start();
  var (asegs, avals) = newRandStrings(n, minLen, maxLen, charSet.Uppercase);
  var astrings = new owned SegString(asegs, avals, st);
  var (bsegs, bvals) = newRandStrings(m, minLen, maxLen, charSet.Uppercase);
  var bstrings = new owned SegString(bsegs, bvals, st);
  t.stop();
  writeln("%t seconds".format(t.elapsed())); stdout.flush(); t.clear();
  writeln("Sorting each array separately..."); stdout.flush();
  t.start();
  var aperm = astrings.argsort();
  var (sasegs, savals) = astrings[aperm];
  var sa = new owned SegString(sasegs, savals, st);
  var bperm = bstrings.argsort();
  var (sbsegs, sbvals) = bstrings[bperm];
  var sb = new owned SegString(sbsegs, sbvals, st);
  t.stop();
  writeln("%t seconds".format(t.elapsed())); stdout.flush(); t.clear();
  writeln("Merging sorted arrays..."); stdout.flush();
  t.start();
  var (cperm, csegs, cvals) = mergeSorted(sa, sb);
  t.stop();
  writeln("%t seconds".format(t.elapsed())); stdout.flush(); t.clear();
  var cstrings = new owned SegString(csegs, cvals, st);
  cstrings.show(5);
  writeln("Result is sorted? >>> ", cstrings.isSorted(), " <<<");
}

config const N = 1_000;
config const TRIALS = 10;
config const M = 100;
config const MINLEN = 1;
config const MAXLEN = 10;

proc main() {
  if testBinarySearch(N, TRIALS) {
    testMerge(N, M, MINLEN, MAXLEN);
  }
}