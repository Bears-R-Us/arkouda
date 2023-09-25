module Broadcast {
  use AryUtil;
  use SymArrayDmapCompat;
  use CommAggregation;
  use BigInteger;
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
    // TODO figure out a way to do memory checking for bigint
    if t != bigint {
      overMemLimit(numBytes(t) * expandedVals.size);
    }
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

  proc broadcast(perm: [?D] int, segs: [?sD] int, segString: borrowed SegString) throws {
    ref offs = segString.offsets.a;
    ref vals = segString.values.a;
    const size = perm.size;
    const high = sD.high;
    var strLens: [sD] int;
    var segLens: [sD] int;
    var expandedLen: int;

    forall (i, off, str_len, seg, seg_len) in zip (sD, offs, strLens, segs, segLens) with (+ reduce expandedLen) {
      if i == high {
        (seg_len, str_len) = (size - seg, vals.size - off);
      } else {
        (seg_len, str_len) = (segs[i+1] - seg, offs[i+1] - off);
      }
      expandedLen += seg_len * str_len;
    }
    var broadDist = broadcast(perm, segs, strLens);
    const offsets = (+ scan broadDist) - broadDist;
    var expandedVals = makeDistArray(expandedLen, uint(8));

    forall (off, str_len, seg, seg_len) in zip(offs, strLens, segs, segLens) with (var valAgg = newDstAggregator(uint(8))) {
      var localizedVals = new lowLevelLocalizingSlice(vals, off..#str_len);
      for i in seg..#seg_len {
        var expValOff = offsets[i];
        for k in 0..#str_len {
          valAgg.copy(expandedVals[expValOff+k], localizedVals.ptr[k]);
        }
      }
    }
    return (expandedVals, offsets);
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
    // TODO figure out a way to do memory checking for bigint
    if t != bigint {
      overMemLimit(numBytes(t) * expandedVals.size);
    }
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
    ref offs = segString.offsets.a;
    ref vals = segString.values.a;
    const high = sD.high;
    var strLens: [sD] int;
    var segLens: [sD] int;
    var expandedLen: int;

    forall (i, off, str_len, seg, seg_len) in zip (sD, offs, strLens, segs, segLens) with (+ reduce expandedLen) {
      if i == high {
        (seg_len, str_len) = (size - seg, vals.size - off);
      } else {
        (seg_len, str_len) = (segs[i+1] - seg, offs[i+1] - off);
      }
      expandedLen += seg_len * str_len;
    }
    var broadDist = broadcast(segs, strLens, size);
    const offsets = (+ scan broadDist) - broadDist;
    var expandedVals = makeDistArray(expandedLen, uint(8));

    forall (off, str_len, seg, seg_len) in zip(offs, strLens, segs, segLens) with (var valAgg = newDstAggregator(uint(8))) {
      var localizedVals = new lowLevelLocalizingSlice(vals, off..#str_len);
      for i in seg..#seg_len {
        var expValOff = offsets[i];
        for k in 0..#str_len {
          valAgg.copy(expandedVals[expValOff+k], localizedVals.ptr[k]);
        }
      }
    }
    return (expandedVals, offsets);
  }
}
