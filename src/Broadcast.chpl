module Broadcast {
  use SymArrayDmap;
  use CommAggregation;
  use SegmentedString;
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
    // The stragegy is to go from the segment domain to the full
    // domain by forming the full derivative and integrating it

    // Compute the sparse derivative (in segment domain) of values
    var diffs: [sD] t;
    forall (i, d, v) in zip(sD, diffs, vals) {
      if i == sD.low {
        d = v;
      } else {
        d = v - vals[i-1];
      }
    }
    // Convert to the dense derivative (in full domain) of values
    var expandedVals: [D] t;
    forall (s, d) in zip(segs, diffs) with (var agg = newDstAggregator(t)) {
      agg.copy(expandedVals[s], d);
    }
    // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
    overMemLimit(numBytes(t) * expandedVals.size);
    // Integrate to recover full values
    expandedVals = (+ scan expandedVals);
    // Permute to the original array order
    var permutedVals: [D] t;
    forall (i, v) in zip(perm, expandedVals) with (var agg = newDstAggregator(t)) {
      agg.copy(permutedVals[i], v);
    }
    return permutedVals;
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
    // The stragegy is to go from the segment domain to the full
    // domain by forming the full derivative and integrating it
    
    // Compute the sparse derivative (in segment domain) of values
    // Treat booleans as integers
    var diffs: [sD] int(8);
    forall (i, d, v) in zip(sD, diffs, vals) {
      if i == sD.low {
        d = v:int(8);
      } else {
        d = v:int(8) - vals[i-1]:int(8);
      }
    }
    // Convert to the dense derivative (in full domain) of values
    var expandedVals: [D] int(8);
    forall (s, d) in zip(segs, diffs) with (var agg = newDstAggregator(int(8))) {
      agg.copy(expandedVals[s], d);
    }
    // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
    overMemLimit(numBytes(int(8)) * expandedVals.size);
    // Integrate to recover full values
    expandedVals = (+ scan expandedVals);
    // Permute to the original array order and convert back to bool
    var permutedVals: [D] bool;
    forall (i, v) in zip(perm, expandedVals) with (var agg = newDstAggregator(bool)) {
      agg.copy(permutedVals[i], v == 1);
    }
    return permutedVals;
  }

  /* 
   * Broadcast a value per segment of a segmented array to the
   * full size of the array. For example, if the segmented array
   * is a compressed sparse row matrix, then expand a row
   * vector such that each nonzero receives its row's value.
   */
  proc broadcast(segs: [?sD] int, vals: [sD] ?t, size: int) throws {
    // The stragegy is to go from the segment domain to the full
    // domain by forming the full derivative and integrating it
    
    // Compute the sparse derivative (in segment domain) of values
    var diffs: [sD] t;
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
    overMemLimit(numBytes(t) * expandedVals.size);
    // Integrate to recover full values
    expandedVals = (+ scan expandedVals);
    return expandedVals;
  }

  proc broadcast(segs: [?sD] int, vals: [sD] bool, size: int) throws {
    // The stragegy is to go from the segment domain to the full
    // domain by forming the full derivative and integrating it
    
    // Compute the sparse derivative (in segment domain) of values
    // Treat booleans as integers
    var diffs: [sD] int(8);
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

  proc broadcast(segs: [?sD] int, segString: borrowed SegString, size: int) throws {
    ref segOff = segString.offsets.a;
    const high = sD.high;
    var strSize: int;
    var diffs: [sD] int;

    forall (i, d, seg, off) in zip (sD, diffs, segs, segOff) with (+ reduce strSize) {
      if i == high {
        strSize += (size - seg) * (segString.nBytes - off);
        d = segString.nBytes - off;
      } else {
        strSize += (segs[i+1] - seg) * (segOff[i+1] - off);
        d = segOff[i+1] - off;
      }
    }

    var broadDist = broadcast(segs, diffs, size);
    var offDiff = makeDistArray(size, int);
    offDiff[1..] = broadDist[..size-2];

    var offsets = (+ scan offDiff);
    var r: [sD] int = 0..segString.size; 
    var ind = broadcast(segs, r, size);
    var expandedVals = makeDistArray(strSize, uint(8));

    forall (i, o, s) in zip(ind, offsets, 0..#size) with (var agg = newDstAggregator(uint(8)), ref vals = segString.values.a) {
      var inds = if i == high then segOff[i]..segString.nBytes-1 else segOff[i]..segOff[i+1]-1;
      var offs = if s == size - 1 then o..strSize-1 else o..offsets[s+1]-1;
      forall (off, idx) in zip(offs, inds) {
        agg.copy(expandedVals[off], vals[idx]);
      }
    }

    return (expandedVals, offsets);
  }
}