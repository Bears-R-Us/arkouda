use TestBase;
use CommAggregation;

config const SIZE = numLocales * here.maxTaskPar;
config const GROUPS = 8;
config const offset = 0;

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
  var diff = res - answers;
  writeCols("grp st size res diff", segments, answers, res, diff);

  if !(&& reduce (res == answers)) {
    writeln(">>> Incorrect result <<<");
  }
}

proc segSum(values:[?vD] ?intype, segments:[?D] int, skipNan=false) throws {
  type t = if intype == bool then int else intype;
  var res: [D] t;
  if (D.size == 0) { return res; }
  // Set reset flag at segment boundaries
  var flagvalues: [vD] (bool, int, t); // = [v in values] (false, v);
  if isFloatType(t) && skipNan {
    forall (fv, val, i) in zip(flagvalues, values, vD) {
      fv = if isnan(val) then (false, i, 0.0) else (false, i, val);
    }
  } else {
    forall (fv, val, i) in zip(flagvalues, values, vD) {
      fv = (false, i, val:t);
    }
  }
  forall s in segments with (var agg = newDstAggregator(bool)) {
    agg.copy(flagvalues[s][0], true);
  }
  // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
  overMemLimit((numBytes(t)+1+8) * flagvalues.size);
  // Scan with custom operator, which resets the bitwise AND
  // at segment boundaries.
  const scanresult = ResettingPlusScanOp scan flagvalues;
  // Read the results from the last element of each segment
  forall (r, s) in zip(res[..D.high-1], segments[D.low+1..]) with (var agg = newSrcAggregator(t)) {
    agg.copy(r, scanresult[s-1](1));
  }
  res[D.high] = scanresult[vD.high](1);
  return res;
}

/* Performs a bitwise sum scan, controlled by a reset flag. While
 * the reset flag is false, the accumulation of values proceeds as 
 * normal. When a true is encountered, the state resets to the
 * identity. */
class ResettingPlusScanOp: ReduceScanOp {
  type eltType;
  /* value is a tuple comprising a flag and the actual result of 
     segmented sum. 

     The meaning of the flag depends on whether it belongs to an 
     array element yet to be scanned or to an element that has 
     already been scanned (including the internal state of a class
     instance doing the scanning). For elements yet to be scanned,
     the flag means "reset to the identity here". For elements that
     have already been scanned, or for internal state, the flag means 
     "there has already been a reset in the computation of this value".
  */
  var value = if eltType == (bool, int, real) then (false, 0, 0.0) else (false, 0, 0);

  proc identity return if eltType == (bool, int, real) then (false, 0, 0.0) else (false, 0, 0);

  proc accumulate(x) {
    // Assume x is an element that has not yet been scanned, and
    // that it comes after the current state.
    const (reset, otherStart, other) = x;
    const (hasReset, start, v) = value;
    // x's reset flag controls whether value gets replaced or combined
    // also update this instance's "hasReset" flag with x's reset flag
    value = (hasReset | reset, min(otherStart, start), if reset then other else (v + other));
  }

  proc accumulateOntoState(ref state, x) {
    // Assume state is an element that has already been scanned,
    // and x is an update from a previous boundary.
    const (xReset, xStart, xVal) = x;
    const (sReset, sStart, sVal) = state;
    const xIsOld = sStart >= xStart;
    var newReset = if xIsOld then sReset else xReset;
    var newStart = if xIsOld then xStart else sStart;
    var newVal = if newReset then (if xIsOld then sVal else xVal) else (sVal + xVal);
    // x's hasReset flag does not matter
    // If state has already encountered a reset, then it should
    // ignore x's value
    state = (newReset, newStart, newVal);
  }

  proc combine(x) {
    // Assume x is an instance that scanned a prior chunk.
    const (xReset, xStart, xVal) = x.value;
    const (sReset, sStart, sVal) = value;
    // Since current instance is absorbing x's history,
    // xHasReset flag should be ORed in.
    // But if current instance has already encountered a reset,
    // then it should ignore x's value.
    const xIsOld = sStart >= xStart;
    var newReset = sReset | xReset;
    var newStart = if xIsOld then xStart else sStart;
    const control = if xIsOld then sReset else xReset;
    var newVal = if control then (if xIsOld then sVal else xVal) else (sVal + xVal);
    value = (newReset, newStart, newVal);
  }

  proc generate() {
    return value;
  }

  proc clone() {
    return new unmanaged ResettingPlusScanOp(eltType=eltType);
  }
}

proc segSum2(values:[?vD] ?intype, segments:[?D] int, skipNan=false) throws {
  type t = if intype == bool then int else intype;
  var res: [D] t;
  if (D.size == 0) { return res; }
  // Set reset flag at segment boundaries
  var flagvalues: [vD] (bool, t); // = [v in values] (false, v);
  if isFloatType(t) && skipNan {
    forall (fv, val) in zip(flagvalues, values) {
      fv = if isnan(val) then (false, 0.0) else (false, val);
    }
  } else {
    forall (fv, val) in zip(flagvalues, values) {
      fv = (false, val:t);
    }
  }
  forall s in segments with (var agg = newDstAggregator(bool)) {
    agg.copy(flagvalues[s][0], true);
  }
  // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
  overMemLimit((numBytes(t)+1) * flagvalues.size);
  // Scan with custom operator, which resets the bitwise AND
  // at segment boundaries.
  // const scanresult = ResettingPlusScanOp scan flagvalues;
  const op = new ResettingPlusScanOp((bool, t));
  const scanresult = flagvalues.nonCommutativeScan(op, vD);
  // Read the results from the last element of each segment
  forall (r, s) in zip(res[..D.high-1], segments[D.low+1..]) with (var agg = newSrcAggregator(t)) {
    agg.copy(r, scanresult[s-1](1));
  }
  res[D.high] = scanresult[vD.high](1);
  return res;
}

/* Performs a bitwise sum scan, controlled by a reset flag. While
 * the reset flag is false, the accumulation of values proceeds as 
 * normal. When a true is encountered, the state resets to the
 * identity. */
class ResettingPlusScanOp2 {
  type eltType;
  /* value is a tuple comprising a flag and the actual result of 
     segmented sum. 

     The meaning of the flag depends on whether it belongs to an 
     array element yet to be scanned or to an element that has 
     already been scanned (including the internal state of a class
     instance doing the scanning). For elements yet to be scanned,
     the flag means "reset to the identity here". For elements that
     have already been scanned, or for internal state, the flag means 
     "there has already been a reset in the computation of this value".
  */
  var value = if eltType == (bool, real) then (false, 0.0) else (false, 0);

  proc identity return if eltType == (bool, real) then (false, 0.0) else (false, 0);

  proc accumulate(x) {
    // Assume x is an element that has not yet been scanned, and
    // that it comes after the current state.
    const (reset, other) = x;
    const (hasReset, v) = value;
    // x's reset flag controls whether value gets replaced or combined
    // also update this instance's "hasReset" flag with x's reset flag
    value = (hasReset | reset, if reset then other else (v + other));
  }

  proc accumulateOntoState(ref state, x) {
    // Assume state is an element that has already been scanned,
    // and x is an update from a previous boundary.
    const (_, other) = x;
    const (hasReset, v) = state;
    // x's hasReset flag does not matter
    // If state has already encountered a reset, then it should
    // ignore x's value
    state = (hasReset, if hasReset then v else (v + other));
  }

  proc combine(x) {
    // Assume x is an instance that scanned a prior chunk.
    const (xHasReset, other) = x.value;
    const (hasReset, v) = value;
    // Since current instance is absorbing x's history,
    // xHasReset flag should be ORed in.
    // But if current instance has already encountered a reset,
    // then it should ignore x's value.
    value = (hasReset | xHasReset, if hasReset then v else (v + other));
  }

  proc generate() {
    return value;
  }

  proc clone() {
    return new unmanaged ResettingPlusScanOp(eltType=eltType);
  }
}

proc BlockArr.nonCommutativeScan(op, dom) where (rank == 1) &&
                                     chpl__scanStateResTypesMatch(op) {

  // The result of this scan, which will be Block-distributed as well
  type resType = op.generate().type;
  var res = dom.buildArray(resType, initElts=!isPOD(resType));

  // Store one element per locale in order to track our local total
  // for a cross-locale scan as well as flags to negotiate reading and
  // writing it.  This domain really wants an easier way to express
  // it...
  use ReplicatedDist;
  const ref targetLocs = this.dsiTargetLocales();
  const elemPerLocDom = {1..1} dmapped Replicated(targetLocs);
  var elemPerLoc: [elemPerLocDom] resType;
  var inputReady$: [elemPerLocDom] sync bool;
  var outputReady$: [elemPerLocDom] sync bool;

  // Fire up tasks per participating locale
  coforall locid in dom.dist.targetLocDom {
    on targetLocs[locid] {
      const myop = op.clone(); // this will be deleted by doiScan()

      // set up some references to our LocBlockArr descriptor, our
      // local array, local domain, and local result elements
      ref myLocArrDesc = locArr[locid];
      ref myLocArr = myLocArrDesc.myElems;
      const ref myLocDom = myLocArr.domain;

      // Compute the local pre-scan on our local array
      var (numTasks, rngs, state, tot) = myLocArr._value.chpl__nonCommutativePreScan(myop, res, myLocDom[dom]);
      if debugBlockScan then
        writeln(locid, ": ", (numTasks, rngs, state, tot));

      // save our local scan total away and signal that it's ready
      elemPerLoc[1] = tot;
      inputReady$[1].writeEF(true);

      // the "first" locale scans the per-locale contributions as they
      // become ready
      if (locid == dom.dist.targetLocDom.low) {
        const metaop = op.clone();

        var next: resType = metaop.identity;
        for locid in dom.dist.targetLocDom {
          const targetloc = targetLocs[locid];
          const locready = inputReady$.replicand(targetloc)[1].readFE();

          // store the scan value and mark that it's ready
          ref locVal = elemPerLoc.replicand(targetloc)[1];
          metaop.accumulateOntoState(locVal, next);
          next = locVal;
          // locVal <=> next;
          outputReady$.replicand(targetloc)[1].writeEF(true);

          // accumulate to prep for the next iteration
          // metaop.accumulateOntoState(next, locVal);
        }
        delete metaop;
      }

      // block until someone tells us that our local value has been updated
      // and then read it
      const resready = outputReady$[1].readFE();
      const myadjust = elemPerLoc[1];
      if debugBlockScan then
        writeln(locid, ": myadjust = ", myadjust);

      // update our state vector with our locale's adjustment value
      for s in state do
        myop.accumulateOntoState(s, myadjust);
      if debugBlockScan then
        writeln(locid, ": state = ", state);

      // have our local array compute its post scan with the globally
      // accurate state vector
      myLocArr._value.chpl__postScan(op, res, numTasks, rngs, state);
      if debugBlockScan then
        writeln(locid, ": ", myLocArr);

      delete myop;
    }
  }
  if isPOD(resType) then res.dsiElementInitializationComplete();

  // delete op;
  return res;
}


// A helper routine to take the first parallel scan over a vector
// yielding the number of tasks used, the ranges computed by each
// task, and the scanned results of each task's scan.  This is
// broken out into a helper function in order to be made use of by
// distributed array scans.
proc DefaultRectangularArr.chpl__nonCommutativePreScan(op, res: [] ?resType, dom) {
  import RangeChunk;
  // Compute who owns what
  const rng = dom.dim(0);
  const numTasks = if __primitive("task_get_serial") then
    1 else _computeNumChunks(rng.sizeAs(int));
  const rngs = RangeChunk.chunks(rng, numTasks);
  if debugDRScan {
    writeln("Using ", numTasks, " tasks");
    writeln("Whose chunks are: ", rngs);
  }

  var state: [rngs.domain] resType;

  // Take first pass over data doing per-chunk scans
  coforall tid in rngs.domain {
    const current: resType;
    const myop = op.clone();
    for i in rngs[tid] {
      ref elem = dsiAccess(i);
      myop.accumulate(elem);
      res[i] = myop.generate();
    }
    state[tid] = res[rngs[tid].high];
    delete myop;
  }

  if debugDRScan {
    writeln("res = ", res);
    writeln("state = ", state);
  }

  // Scan state vector itself
  const metaop = op.clone();
  var next: resType = metaop.identity;
  for i in rngs.domain {
    // state[i] <=> next;
    // metaop.accumulateOntoState(next, state[i]);
    metaop.accumulateOntoState(state[i], next);
    next = state[i];
  }
  delete metaop;
  if debugDRScan then
    writeln("state = ", state);

  return (numTasks, rngs, state, next);
}

// How many tasks should be spawned to service numElems elements.
proc _computeNumChunks(numElems): int {
  // copy some machinery from DefaultRectangularDom
  var numTasks = if dataParTasksPerLocale==0
                 then here.maxTaskPar
                 else dataParTasksPerLocale;
  var ignoreRunning = dataParIgnoreRunningTasks;
  var minIndicesPerTask = dataParMinGranularity;
  var numChunks = _computeNumChunks(numTasks, ignoreRunning,
                                    minIndicesPerTask, numElems);
  return numChunks;
}

// returns 0 if no numElems <= 0
proc _computeNumChunks(maxTasks, ignoreRunning, minSize, numElems): int {
  if numElems <= 0 then
    return 0;

  type EC = uint; // type for element counts
  const unumElems = numElems:EC;
  var numChunks = maxTasks:int;
  if !ignoreRunning {
    const otherTasks = here.runningTasks() - 1; // don't include self
    numChunks = if otherTasks < maxTasks
      then (maxTasks-otherTasks):int
      else 1;
  }

  if minSize > 0 then
    // This is approximate
    while (unumElems < (minSize*numChunks):EC) && (numChunks > 1) {
        numChunks -= 1;
    }

  if numChunks:EC > unumElems then numChunks = unumElems:int;

  return numChunks;
}
