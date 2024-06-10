module Broadcast {
  use AryUtil;
  use SymArrayDmap;
  use CommAggregation;
  use BigInteger;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Reflection;
  use Logging;
  use ServerConfig;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const brLogger = new Logger(logLevel, logChannel);

  /* 
   * Broadcast a value per segment of a segmented array to the
   * original ordering of the precursor array. For example, if
   * the original array was sorted and grouped, resulting in 
   * groups defined by <segs>, and if <vals> contains group labels,
   * then return the array of group labels corresponding to the 
   * original array. Intended to be used with arkouda.GroupBy.
   */
  proc broadcast(perm: [?D] int, segs: [?sD] int, vals: [sD] ?t) throws {
    if sD.size == 0 {
      // early out if size 0
      return makeDistArray(D.size, t);
    }
    // The stragegy is to go from the segment domain to the full
    // domain by forming the full derivative and integrating it
    var keepSegs = makeDistArray(sD, bool);
    [(k, s, i) in zip(keepSegs, segs, sD)] if i < sD.high { k = (segs[i+1] != s); }
    keepSegs[sD.high] = true;

    const numKeep = + reduce keepSegs;

    if numKeep == sD.size {
      // Compute the sparse derivative (in segment domain) of values
      var diffs = makeDistArray(sD, t);
      forall (i, d, v) in zip(sD, diffs, vals) {
        if i == sD.low {
          d = v;
        } else {
          d = v - vals[i-1];
        }
      }
      // Convert to the dense derivative (in full domain) of values
      var expandedVals = makeDistArray(D, t);
      forall (s, d) in zip(segs, diffs) with (var agg = newDstAggregator(t)) {
        agg.copy(expandedVals[s], d);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      // TODO figure out a way to do memory checking for bigint
      if t != bigint {
        overMemLimit(numBytes(t) * expandedVals.size);
      }
      // Integrate to recover full values
      expandedVals = (+ scan expandedVals);
      // Permute to the original array order
      var permutedVals = makeDistArray(D, t);
      forall (i, v) in zip(perm, expandedVals) with (var agg = newDstAggregator(t)) {
        agg.copy(permutedVals[i], v);
      }
      return permutedVals;
    }
    else {
      // boolean indexing into segs and vals
      const iv = + scan keepSegs - keepSegs;
      const kD = makeDistDom(numKeep);
      var compressedSegs: [kD] int;
      var compressedVals: [kD] t;
      forall (i, keep, seg, val) in zip(sD, keepSegs, segs, vals) with (var segAgg = newDstAggregator(int),
                                                                        var valAgg = newDstAggregator(t)) {
        if keep {
          segAgg.copy(compressedSegs[iv[i]], seg);
          valAgg.copy(compressedVals[iv[i]], val);
        }
      }
      // Compute the sparse derivative (in segment domain) of values
      var diffs = makeDistArray(kD, t);
      forall (i, d, v) in zip(kD, diffs, compressedVals) {
        if i == sD.low {
          d = v;
        } else {
          d = v - compressedVals[i-1];
        }
      }
      // Convert to the dense derivative (in full domain) of values
      var expandedVals = makeDistArray(D, t);
      forall (s, d) in zip(compressedSegs, diffs) with (var agg = newDstAggregator(t)) {
        agg.copy(expandedVals[s], d);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      // TODO figure out a way to do memory checking for bigint
      if t != bigint {
        overMemLimit(numBytes(t) * expandedVals.size);
      }
      // Integrate to recover full values
      expandedVals = (+ scan expandedVals);
      // Permute to the original array order
      var permutedVals = makeDistArray(D, t);
      forall (i, v) in zip(perm, expandedVals) with (var agg = newDstAggregator(t)) {
        agg.copy(permutedVals[i], v);
      }
      return permutedVals;
    }
  }

  /* 
   * Broadcast a value per segment of a segmented array to the
   * original ordering of the precursor array. For example, if
   * the original array was sorted and grouped, resulting in 
   * groups defined by <segs>, and if <vals> contains group labels,
   * then return the array of group labels corresponding to the 
   * original array. Intended to be used with arkouda.GroupBy.
   */
  proc broadcast(perm: [?D] int, segs: [?sD] int, vals: [sD] bool) throws {
    if sD.size == 0 {
      // early out if size 0
      return makeDistArray(D.size, bool);
    }
    // The stragegy is to go from the segment domain to the full
    // domain by forming the full derivative and integrating it    
    var keepSegs = makeDistArray(sD, bool);
    [(k, s, i) in zip(keepSegs, segs, sD)] if i < sD.high { k = (segs[i+1] != s); }
    keepSegs[sD.high] = true;

    const numKeep = + reduce keepSegs;

    if numKeep == sD.size {
      // Compute the sparse derivative (in segment domain) of values
      // Treat booleans as integers
      var diffs = makeDistArray(sD, int(8));
      forall (i, d, v) in zip(sD, diffs, vals) {
        if i == sD.low {
          d = v:int(8);
        } else {
          d = v:int(8) - vals[i-1]:int(8);
        }
      }
      // Convert to the dense derivative (in full domain) of values
      var expandedVals = makeDistArray(D, int(8));
      forall (s, d) in zip(segs, diffs) with (var agg = newDstAggregator(int(8))) {
        agg.copy(expandedVals[s], d);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int(8)) * expandedVals.size);
      // Integrate to recover full values
      expandedVals = (+ scan expandedVals);
      // Permute to the original array order and convert back to bool
      var permutedVals = makeDistArray(D, bool);
      forall (i, v) in zip(perm, expandedVals) with (var agg = newDstAggregator(bool)) {
        agg.copy(permutedVals[i], v == 1);
      }
      return permutedVals;
    }
    else {
      // boolean indexing into segs and vals
      const iv = + scan keepSegs - keepSegs;
      const kD = makeDistDom(numKeep);
      var compressedSegs: [kD] int;
      var compressedVals: [kD] bool;
      forall (i, keep, seg, val) in zip(sD, keepSegs, segs, vals) with (var segAgg = newDstAggregator(int),
                                                                        var valAgg = newDstAggregator(bool)) {
        if keep {
          segAgg.copy(compressedSegs[iv[i]], seg);
          valAgg.copy(compressedVals[iv[i]], val);
        }
      }

      // Compute the sparse derivative (in segment domain) of values
      // Treat booleans as integers
      var diffs = makeDistArray(kD, int(8));
      forall (i, d, v) in zip(kD, diffs, compressedVals) {
        if i == kD.low {
          d = v:int(8);
        } else {
          d = v:int(8) - compressedVals[i-1]:int(8);
        }
      }
      // Convert to the dense derivative (in full domain) of values
      var expandedVals = makeDistArray(D, int(8));
      forall (s, d) in zip(compressedSegs, diffs) with (var agg = newDstAggregator(int(8))) {
        agg.copy(expandedVals[s], d);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int(8)) * expandedVals.size);
      // Integrate to recover full values
      expandedVals = (+ scan expandedVals);
      // Permute to the original array order and convert back to bool
      var permutedVals = makeDistArray(D, bool);
      forall (i, v) in zip(perm, expandedVals) with (var agg = newDstAggregator(bool)) {
        agg.copy(permutedVals[i], v == 1);
      }
      return permutedVals;
    }
  }

  /* 
   * Broadcast a value per segment of a segmented array to the
   * full size of the array. For example, if the segmented array
   * is a compressed sparse row matrix, then expand a row
   * vector such that each nonzero receives its row's value.
   */
  proc broadcast(segs: [?sD] int, vals: [sD] ?t, size: int) throws {
    if sD.size == 0 {
      // early out if size 0
      return makeDistArray(size, t);
    }
    // The stragegy is to go from the segment domain to the full
    // domain by forming the full derivative and integrating it
    var keepSegs = makeDistArray(sD, bool);
    [(k, s, i) in zip(keepSegs, segs, sD)] if i < sD.high { k = (segs[i+1] != s); }
    keepSegs[sD.high] = true;

    const numKeep = + reduce keepSegs;
    if numKeep == sD.size {
      // Compute the sparse derivative (in segment domain) of values
      var diffs = makeDistArray(sD, t);
      forall (i, d, v) in zip(sD, diffs, vals) {
        if i == sD.low {
          d = v;
        } else {
          d = v - vals[i-1];
        }
      }
      // Convert to the dense derivative (in full domain) of values
      var expandedVals = makeDistArray(size, t);
      forall (s, d) in zip(segs, diffs) with (var agg = newDstAggregator(t)) {
        agg.copy(expandedVals[s], d);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      // TODO figure out a way to do memory checking for bigint
      if t != bigint {
        overMemLimit(numBytes(t) * expandedVals.size);
      }
      // Integrate to recover full values
      expandedVals = (+ scan expandedVals);
      return expandedVals;
    }
    else {
      // boolean indexing into segs and vals
      const iv = + scan keepSegs - keepSegs;
      const kD = makeDistDom(numKeep);
      var compressedSegs: [kD] int;
      var compressedVals: [kD] t;
      forall (i, keep, seg, val) in zip(sD, keepSegs, segs, vals) with (var segAgg = newDstAggregator(int),
                                                                        var valAgg = newDstAggregator(t)) {
        if keep {
          segAgg.copy(compressedSegs[iv[i]], seg);
          valAgg.copy(compressedVals[iv[i]], val);
        }
      }
      // Compute the sparse derivative (in segment domain) of values
      var diffs = makeDistArray(kD, t);
      forall (i, d, v) in zip(kD, diffs, compressedVals) {
        if i == kD.low {
          d = v;
        } else {
          d = v - compressedVals[i-1];
        }
      }
      // Convert to the dense derivative (in full domain) of values
      var expandedVals = makeDistArray(size, t);
      forall (s, d) in zip(compressedSegs, diffs) with (var agg = newDstAggregator(t)) {
        agg.copy(expandedVals[s], d);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      // TODO figure out a way to do memory checking for bigint
      if t != bigint {
        overMemLimit(numBytes(t) * expandedVals.size);
      }
      // Integrate to recover full values
      expandedVals = (+ scan expandedVals);
      return expandedVals;
    }
  }

  proc broadcast(segs: [?sD] int, vals: [sD] bool, size: int) throws {
    if sD.size == 0 {
      // early out if size 0
      return makeDistArray(size, bool);
    }
    // The stragegy is to go from the segment domain to the full
    // domain by forming the full derivative and integrating it
    var keepSegs = makeDistArray(sD, bool);
    [(k, s, i) in zip(keepSegs, segs, sD)] if i < sD.high { k = (segs[i+1] != s); }
    keepSegs[sD.high] = true;

    const numKeep = + reduce keepSegs;
    if numKeep == sD.size {
      // Compute the sparse derivative (in segment domain) of values
      // Treat booleans as integers
      var diffs = makeDistArray(sD, int(8));
      forall (i, d, v) in zip(sD, diffs, vals) {
        if i == sD.low {
          d = v:int(8);
        } else {
          d = v:int(8) - vals[i-1]:int(8);
        }
      }
      // Convert to the dense derivative (in full domain) of values
      var expandedVals = makeDistArray(size, int(8));
      forall (s, d) in zip(segs, diffs) with (var agg = newDstAggregator(int(8))) {
        agg.copy(expandedVals[s], d);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int(8)) * expandedVals.size);
      // Integrate to recover full values
      expandedVals = (+ scan expandedVals);
      return (expandedVals == 1);
    }
    else {
      // boolean indexing into segs and vals
      const iv = + scan keepSegs - keepSegs;
      const kD = makeDistDom(numKeep);
      var compressedSegs: [kD] int;
      var compressedVals: [kD] bool;
      forall (i, keep, seg, val) in zip(sD, keepSegs, segs, vals) with (var segAgg = newDstAggregator(int),
                                                                        var valAgg = newDstAggregator(bool)) {
        if keep {
          segAgg.copy(compressedSegs[iv[i]], seg);
          valAgg.copy(compressedVals[iv[i]], val);
        }
      }

      // Compute the sparse derivative (in segment domain) of values
      // Treat booleans as integers
      var diffs = makeDistArray(kD, int(8));
      forall (i, d, v) in zip(kD, diffs, compressedVals) {
        if i == kD.low {
          d = v:int(8);
        } else {
          d = v:int(8) - compressedVals[i-1]:int(8);
        }
      }
      // Convert to the dense derivative (in full domain) of values
      var expandedVals = makeDistArray(size, int(8));
      forall (s, d) in zip(compressedSegs, diffs) with (var agg = newDstAggregator(int(8))) {
        agg.copy(expandedVals[s], d);
      }
      // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
      overMemLimit(numBytes(int(8)) * expandedVals.size);
      // Integrate to recover full values
      expandedVals = (+ scan expandedVals);
      return (expandedVals == 1);
    }
  }
}
