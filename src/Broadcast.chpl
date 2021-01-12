module Broadcast {
  use SymArrayDmap;
  use CommAggregation;

  /* 
   * Broadcast a value per segment of a segmented array to the
   * original ordering of the precursor array. For example, if
   * the original array was sorted and grouped, resulting in 
   * groups defined by <segs>, and if <vals> contains group labels,
   * then return the array of group labels corresponding to the 
   * original array. Intended to be used with arkouda.GroupBy.
   */
  proc broadcast(perm: [?D] int, segs: [?sD] int, vals: [sD] ?t) {
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
  proc broadcast(perm: [?D] int, segs: [?sD] int, vals: [sD] bool) {
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
  proc broadcast(segs: [?sD] int, vals: [sD] ?t, size: int) {
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
    // Integrate to recover full values
    expandedVals = (+ scan expandedVals);
    return expandedVals;
  }

  proc broadcast(segs: [?sD] int, vals: [sD] bool, size: int) {
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
    // Integrate to recover full values
    expandedVals = (+ scan expandedVals);
    return (expandedVals == 1);
  }
}