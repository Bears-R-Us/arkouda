use TestBase;
use CommAggregation;

config const NINPUTS = 10_000;
config const DEBUG = false;
config const TOY = false;

proc hexlify(a) {
  var fmt = "";
  for x in a {
    fmt += "%xi ".format(x);
  }
  return fmt;
}

proc toy() {
  var answer = [0, 1, 3, 4, 0xc, 0x1c, 0x20, 0x60, 0xe0, 0x100, 0x300, 0x700];
  var values = [0, 1, 2, 4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x100, 0x200, 0x400];
  var segments = [0, 3, 6, 9];
  var flags: [values.domain] bool = false;
  for s in segments do flags[s] = true;
  var toscan = [(f, v) in zip(flags, values)] (f, v);
  var scanned = ResettingOrScanOp scan toscan;
  var resval = [s in scanned] s[1];
  var resflag = [s in scanned] s[0];
  writeln("segs = ", segments);
  writeln("vals = %t".format(hexlify(values)));
  writeln("scan = %t".format(hexlify(resval)));
  writeln("       %t".format(hexlify(resflag:int)));
  writeln("Correct? %t".format(&& reduce (answer == resval)));
}

proc makeArrays(N: int) {
  const sD = makeDistDom(N/7);
  const lengths: [sD] int = [i in sD] (i % 13) + 1;
  var segments = (+ scan lengths) - lengths;
  const size = + reduce lengths;
  const vD = makeDistDom(size);
  // Set up values so that each segment contains 1..lengths[s]
  // Init with 1, except at segment boundaries decrement
  // by previous segment's length-1 to bring back to 1.
  var values: [vD] int = 1;
  forall (s, l) in zip(segments[sD.low+1..], lengths[..sD.high-1]) with (var agg = newDstAggregator(int)) {
    agg.copy(values[s], 1-l);
  }
  values = + scan values;
  // Because each segment is a range, its OR is easy
  var answers: [sD] int;
  answers |= (lengths >= 1):int;
  answers |= (lengths >= 2):int << 1;
  answers |= (lengths >= 4):int << 2;
  answers |= (lengths >= 8): int << 3;
  return (segments, values, answers);
}

proc main() {
  if TOY {
    toy();
    return;
  }
  var d: Diags;
  var (segments, values, answers) = makeArrays(NINPUTS);
  if DEBUG {
    writeln("segs = ", segments);
    writeln("vals = ", hexlify(values));
    writeln("ans  = ", hexlify(answers));
  }
  d.start();
  var res1 = segOr1(values, segments);
  d.stop("Method 1 (on startLocale): ");
  if !(&& reduce (res1 == answers)) {
    writeln(">>> Incorrect result from Method 1 <<<");
    if DEBUG {
      writeln("res1 = ", hexlify(res1));
    }
  }
  d.start();
  var res2 = segOr2(values, segments);
  d.stop("Method 2 (custom scan class): ");
  if !(&& reduce (res2 == answers)) {
    writeln(">>> Incorrect result from Method 2 <<<");
    if DEBUG {
      writeln("res2 = ", hexlify(res2));
    }
  }
}
  
proc segOr1(values:[?vD] int, segments:[?D] int): [D] int {
  // Bitwise OR does not have an inverse, so this cannot be
  // done with a scan. Each segment's values must be reduced
  // separately.
  var res: [D] int;
  if (D.size == 0) { return res; }
  forall (i, s, r) in zip(D, segments, res) {
    // Find segment end
    var e: int;
    if i < D.high {
      e = segments[i+1] - 1;
    } else {
      e = vD.high;
    }
    // Run computation on locale where segment values start
    // At most, numLocales-1 segments will have to get remote values
    // Some results will have to be sent to remote locales
    ref start = values[s];
    const startLocale = start.locale.id;
    on startLocale {
      r = | reduce values[s..e];
    }
  }
  return res;
}

proc segOr2(values:[?vD] int, segments:[?D] int): [D] int {
  var res: [D] int;
  var flagvalues: [vD] (bool, int) = [v in values] (false, v);
  forall s in segments with (var agg = newDstAggregator(bool)) {
    agg.copy(flagvalues[s][0], true);
  }
  const scanresult = ResettingOrScanOp scan flagvalues;
  forall (r, s) in zip(res[..D.high-1], segments[D.low+1..]) with (var agg = newSrcAggregator(int)) {
    agg.copy(r, scanresult[s-1](1));
  }
  res[D.high] = scanresult[vD.high](1);
  return res;
}

class ResettingOrScanOp: ReduceScanOp {
  type eltType;
  /* value is a tuple comprising a flag and the actual result of 
     segmented bitwise OR. 

     The meaning of the flag depends on
     whether it belongs to an array element yet to be scanned or 
     to an element that has already been scanned (or the state of
     an instance doing the scanning). For elements yet to be scanned,
     the flag means "reset to the identity here". For elements that
     have already been scanned, or for internal state, the flag means 
     "there has already been a reset in the computation of this value".
     */
  var value = (false, 0);

  proc identity return (false, 0);

  proc accumulate(x) {
    // Assume x is an element that has not yet been scanned, and
    // that it comes after the current state.
    const (reset, other) = x;
    const (hasReset, v) = value;
    // x's reset flag controls whether value gets replaced or combined
    // also update this instance's "hasReset" flag with x's reset flag
    value = (hasReset | reset, if reset then other else (v | other));
  }

  proc accumulateOntoState(ref state, x) {
    // Assume state is an element that has already been scanned,
    // and x is an update from a previous boundary.
    const (_, other) = x;
    const (hasReset, v) = state;
    // x's hasReset flag does not matter
    // If state has already encountered a reset, then it should
    // ignore x's value
    state = (hasReset, if hasReset then v else (v | other));
  }

  proc combine(x) {
    // Assume x is an instance that scanned a prior chunk.
    const (xHasReset, other) = x.value;
    const (hasReset, v) = value;
    // Since current instance is absorbing x's history,
    // xHasReset flag should be ORed in.
    // But if current instance has already encountered a reset,
    // then it should ignore x's value.
    value = (hasReset | xHasReset, if hasReset then v else (v | other));
  }

  proc generate() {
    return value;
  }

  proc clone() {
    return new unmanaged ResettingOrScanOp(eltType=eltType);
  }
}
