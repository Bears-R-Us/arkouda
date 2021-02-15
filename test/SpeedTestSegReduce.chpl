use TestBase;
use CommAggregation;

config const NINPUTS = 100_000;

proc makeArrays(N: int) {
  const sD = makeDistDom(N/8);
  const lengths: [sD] int = [i in sD] (i % 15) + 1;
  var segments = (+ scan lengths) - lengths;
  const size = + reduce lengths;
  const vD = makeDistDom(size);
  var values: [vD] int = 1;
  forall (s, l) in zip(segments, lengths) with (var agg = newDstAggregator(int)) {
    agg.copy(values[s], 1-l);
  }
  values = + scan values;
  var answers: [sD] int;
  answers |= (lengths >= 1):int;
  answers |= (lengths >= 3):int << 1;
  answers |= (lengths >= 7):int << 2;
  answers |= (lengths >= 15): int << 3;
  return (segments, values, answers);
}

proc main() {
  var d: Diags;
  var (segments, values, answers) = makeArrays(NINPUTS);
  d.start();
  var res1 = segOr1(values, segments);
  d.stop("Method 1 (on startLocale): ", printTime=true);
  if (&& reduce (res1 == answers)) {
    writeln(">>> Incorrect result from Method 1 <<<");
  }
  d.start();
  var res2 = segOr2(values, segments);
  d.stop("Method 2 (custom scan class): ", printTime=true);
  if (&& reduce (res2 == answers)) {
    writeln(">>> Incorrect result from Method 2 <<<");
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
      e = segments[i+1];
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
    agg.copy(r, scanresult[s-1]);
  }
  res[D.high] = scanresult[vD.high];
  return res;
}

class ResettingOrScanOp: ReduceScanOp {
  type eltType;
  var value = (false, 0);

  proc identity return (false, 0);

  proc accumulate(x) {
    const (reset, other) = x;
    const (_, v) = value;
    value = (reset, if reset then other else (v | other));
  }

  proc accumulateOntoState(ref state, x) {
    const (reset, other) = x;
    const (_, v) = state;
    state = (reset, if reset then other else (v | other));
  }

  proc combine(x) {
    const (reset, other) = x.value;
    const (_, v) = value;
    value = (reset, if reset then other else (v | other));
  }

  proc generate() {
    const (reset, val) = value;
    return val;
  }

  proc clone() {
    return new unmanaged ResettingOrScanOp(eltType);
  }
}