use RandArray;
use MultiTypeSymbolTable;
use SegmentedArray;
use Time;

config const N: int = 10_000;
config const MINLEN: int = 6;
config const MAXLEN: int = 30;
config const SUBSTRING: string = "hi";
config const DEBUG = false;

proc make_strings(substr, n, minLen, maxLen, characters, mode, st) {
  const nb = substr.numBytes;
  const sbytes: [0..#nb] uint(8) = for b in substr.chpl_bytes() do b;
  var (segs, vals) = newRandStringsUniformLength(n, minLen, maxLen, characters);
  var strings = new owned SegString(segs, vals, st);
  var lengths = strings.getLengths() - 1;
  var r: [segs.domain] int;
  fillInt(r, 0, 100);
  var splits = (r % 3);
  /* var present = (r < 5) & (lengths >= nb); */
  /* present[present.domain.high] = true; */
  /* present[present.domain.low] = true; */
  forall (rn, o, l, s) in zip(r, segs, lengths, splits) {
    if l >= nb {
      if (s == 1) || ((s == 2) && (l < 2*(nb+1))) {
        if l == nb {
          vals[{o..#nb}] = sbytes;
        } else {
          const i = rn % (l - nb);
          vals[{(o+i)..#nb}] = sbytes;
        }
      } else if s == 2 {
        const i = rn % ((l/2) - nb);
        vals[{(o+i)..#nb}] = sbytes;
        const j = i + (l/2);
        vals[{(o+j)..#nb}] = sbytes;
      }
    }
  }
  var strings2 = new owned SegString(segs, vals, st);
  return (splits, strings2);
}

proc testSplit(substr:string, n:int, minLen:int, maxLen:int, characters:charSet = charSet.Uppercase, mode: SearchMode = SearchMode.contains) throws {
  var st = new owned SymTab();
  var t = new Timer();
  writeln("Generating random strings..."); stdout.flush();
  t.start();
  var (answer, strings) = make_strings(substr, n, minLen, maxLen, characters, mode, st);
  t.stop();
  writeln("%t seconds".format(t.elapsed())); stdout.flush(); t.clear();
  writeln("Splitting..."); stdout.flush();
  t.start();
  var (numSplits, results) = strings.split(substr, 2);
  t.stop();
  writeln("%t seconds".format(t.elapsed())); stdout.flush();
  for (i, r) in zip(1..results.size, results) {
    writeln("Split %i".format(i));
    var str = new owned SegString(r.splitOffsets, r.splitValues, st);
    str.show();
  }
  writeln("Found correct number of splits? >>> ", && reduce (numSplits == answer), " <<<");
  /* var isMissing = (answer & !truth); */
  /* var missing = + reduce isMissing; */
  /* var isExtra = (truth & !answer); */
  /* var extra = + reduce isExtra; */
  /* if (missing == 0) && (extra == 0) { */
  /*   writeln("Perfect match"); */
  /* } else { */
  /*   writeln("%t missing answers; %t extras".format(missing, extra)); */
  /*   if DEBUG { */
  /*     writeln("missing:"); */
  /*     var minds = + scan isMissing; */
  /*     for i in 1..5 { */
  /*       var (blah, idx) = maxloc reduce zip((minds == i), minds.domain); */
  /*       if !blah { */
  /*         break; */
  /*       } */
  /*       writeln(); */
  /*       if idx > minds.domain.low { */
  /*         writeln(strings[idx-1]); */
  /*       } */
  /*       writeln(strings[idx]); */
  /*       if idx < minds.domain.high { */
  /*         writeln(strings[idx+1]); */
  /*       } */
  /*     } */
  /*     writeln("extras:"); */
  /*     var einds = + scan isExtra; */
  /*     for i in 1..5 { */
  /*       var (blah, idx) = maxloc reduce zip((einds == i), einds.domain); */
  /*       if !blah { */
  /*         break; */
  /*       } */
  /*       writeln(); */
  /*       if idx > einds.domain.low { */
  /*         writeln(strings[idx-1]); */
  /*       } */
  /*       writeln(strings[idx]); */
  /*       if idx < einds.domain.high { */
  /*         writeln(strings[idx+1]); */
  /*       } */
  /*     } */
  /*   } */
  /* } */
}

proc main() {
  for mode in (SearchMode.contains, SearchMode.startsWith, SearchMode.endsWith) {
    writeln("\n", mode);
    try! testSplit(SUBSTRING, N, MINLEN, MAXLEN, mode=mode);
  }
}
  
