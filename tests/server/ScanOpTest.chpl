/*
 * The purpose of this test was to reproduce a bug that only occurs
 * with multiple locales and a large number of cores per locale, so
 * it was not caught by the CI.

 * To trigger those conditions on a single locale, chapel must be
 * built in multi-locale mode (e.g. CHPL_COMM=gasnet) and this
 * program must be run in an oversubscribed configuration, e.g.:
 *
 * CHPL_RT_OVERSUBSCRIBED=yes CHPL_RT_NUM_THREADS_PER_LOCALE=8 test-bin/ScanOpTest -nl 4 --SIZE=32
 */

use TestBase;
use CommAggregation;
use ReductionMsg;

config const SIZE = numLocales * here.maxTaskPar;
config const GROUPS = min(SIZE, 8);
config const offset = 0;
config const DEBUG = false;

proc makeArrays() {
  const sD = makeDistDom(GROUPS);
  const D = makeDistDom(SIZE);
  var keys: [D] int;
  var segs: [sD] int;
  forall (i, k) in zip(D, keys) {
    var key = (i - offset) / (SIZE / GROUPS);
    if key < 0 {
      k = 0;
    } else if key >= GROUPS {
      k = GROUPS - 1;
    } else {
      k = key;
      if ((i - offset) % (SIZE / GROUPS)) == 0 {
        segs[key] = i;
      }
    }
  }
  segs[0] = 0;
  var ones: [D] int = 1;
  var ans: [sD] int;
  for g in sD {
    ans[g] = + reduce (keys == g);
  }
  return (keys, segs, ones, ans);
}

proc writeCols(names: string, a:[?D] int, b: [D] int, c: [D] int, d: [D] int) {
  writeln(names);
  for i in D {
    var line = "%2i %3i %3i %3i %3i".format(i, a[i], b[i], c[i], d[i]);
    writeln(line);
  }
}

proc main() {
  const (keys, segments, values, answers) = makeArrays();
  var res = segSum(values, segments);
  if DEBUG {
    var diff = res - answers;
    writeCols("grp st size res diff", segments, answers, res, diff);
  }

  if !(&& reduce (res == answers)) {
    writeln(">>> Incorrect result <<<");
  }
}