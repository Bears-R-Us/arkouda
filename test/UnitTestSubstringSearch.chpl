use RandArray;
use MultiTypeSymbolTable;
use SegmentedArray;
use Time;

config const N: int = 10_000;
config const MINLEN: int = 1;
config const MAXLEN: int = 20;
config const SUBSTRING: string = "HI";

proc test_search(substr:string, n:int, minLen:int, maxLen:int, characters:charSet = charSet.Uppercase, mode: SearchMode = SearchMode.contains) throws {
  var st = new owned SymTab();
  var t = new Timer();
  writeln("Generating random strings..."); stdout.flush();
  t.start();
  var (segs, vals) = newRandStrings(n, minLen, maxLen, characters);
  var strings = new owned SegString(segs, vals, st);
  t.stop();
  writeln("%t seconds".format(t.elapsed())); stdout.flush(); t.clear();
  writeln("Searching for substring..."); stdout.flush();
  t.start();
  var truth = strings.substringSearch(substr, mode);
  t.stop();
  writeln("%t seconds".format(t.elapsed())); stdout.flush();
  var nFound = + reduce truth;
  writeln("Found %t strings containing %s".format(nFound, substr)); stdout.flush();
  if nFound > 0 {
    var (mSegs, mVals) = strings[truth];
    var matches = new owned SegString(mSegs, mVals, st);
    matches.show(5);
  }
}

proc main() {
  try! test_search(SUBSTRING, N, MINLEN, MAXLEN, mode=SearchMode.contains);
}
  
