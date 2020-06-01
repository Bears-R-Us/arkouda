use TestBase;

use Merge;
use RadixSortLSD;
use Random;

proc testBinarySearch(size, trials) {
  var a = makeDistArray(size, int);
  fillRandom(a);
  var b = radixSortLSD_keys(a);
  var R = new RandomStream(int);
  const fixed = [min(int), b[0], b[size-1], max(int)];
  const ans = [0, 0, size-1, size];
  for (i, f, an) in zip(0..3, fixed, ans) {
    var hit = binarySearch(b, f);
    if hit != an {
      writeln("Binary search failed on fixed round ", i); stdout.flush();
      writeln("x = %t, res = %t, b[res] = %t, ans = %t".format(f, hit, b[hit], an)); stdout.flush();
      return false;
    }
  }
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
  var (asegs, avals) = newRandStringsUniformLength(n, minLen, maxLen, charSet.Uppercase);
  var astrings = new owned SegString(asegs, avals, st);
  var (bsegs, bvals) = newRandStringsUniformLength(m, minLen, maxLen, charSet.Uppercase);
  var bstrings = new owned SegString(bsegs, bvals, st);
  t.stop();
  writeln("%t seconds".format(t.elapsed())); stdout.flush(); t.clear();
  writeln("Sorting each array separately..."); stdout.flush();
  t.start();
  var aperm = astrings.argsort();
  var (sasegs, savals) = astrings[aperm];
  var sa = new shared SegString(sasegs, savals, st);
  var bperm = bstrings.argsort();
  var (sbsegs, sbvals) = bstrings[bperm];
  var sb = new shared SegString(sbsegs, sbvals, st);
  t.stop();
  writeln("%t seconds".format(t.elapsed())); stdout.flush(); t.clear();
  writeln("Merging sorted arrays..."); stdout.flush();
  t.start();
  var (cperm, csegs, cvals) = mergeSorted(sa, sb);
  t.stop();
  writeln("%t seconds".format(t.elapsed())); stdout.flush(); t.clear();
  var cstrings = new owned SegString(csegs, cvals, st);
  cstrings.show(5);
  var diff = cstrings.ediff();
  var sorted = && reduce (diff >= 0);
  writeln("Result is sorted? >>> ", sorted, " <<<");
  if !sorted {
    var pos = 0;
    for i in 0..#5 {
      while (pos < cstrings.size) && (diff[pos] <= 0) {
        pos += 1;
      }
      writeln("\n%i: %i\n%s\n%s".format(pos, diff[pos], cstrings[pos], cstrings[pos+1])); stdout.flush();
      pos += 1;
    }
  }
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
