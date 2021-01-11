module Broadcast {
  use SymArrayDmap;
  use CommAggregation;

  proc broadcast(perm: [?D] int, segs: [?sD] int, vals: [sD] ?t) {
    var expandedVals: [D] t;
    var diffs: [sD] t;
    forall (i, d, v) in zip(sD, diffs, vals) {
      if i == sD.low {
        d = v;
      } else {
        d = v - vals[i-1];
      }
    }
    forall (s, d) in zip(segs, diffs) with (var agg = newDstAggregator(t)) {
      agg.copy(expandedVals[s], d);
    }
    expandedVals = (+ scan expandedVals);
    var permutedVals: [D] t;
    forall (i, v) in zip(perm, expandedVals) with (var agg = newDstAggregator(t)) {
      agg.copy(permutedVals[i], v);
    }
    return permutedVals;
  }

  proc broadcast(perm: [?D] int, segs: [?sD] int, vals: [sD] bool) {
    var expandedVals: [D] uint(8);
    var diffs: [sD] uint(8);
    forall (i, d, v) in zip(sD, diffs, vals) {
      if i == sD.low {
        d = v:uint(8);
      } else {
        d = v:uint(8) - vals[i-1]:uint(8);
      }
    }
    forall (s, d) in zip(segs, diffs) with (var agg = newDstAggregator(uint(8))) {
      agg.copy(expandedVals[s], d);
    }
    expandedVals = (+ scan expandedVals);
    var permutedVals: [D] bool;
    forall (i, v) in zip(perm, expandedVals) with (var agg = newDstAggregator(bool)) {
      agg.copy(permutedVals[i], v == 1);
    }
    return permutedVals;
  }

  proc broadcast(segs: [?sD] int, vals: [sD] ?t, size: int) {
    var expandedVals = makeDistArray(size, t);
    var diffs: [sD] t;
    forall (i, d, v) in zip(sD, diffs, vals) {
      if i == sD.low {
        d = v;
      } else {
        d = v - vals[i-1];
      }
    }
    forall (s, d) in zip(segs, diffs) with (var agg = newDstAggregator(t)) {
      agg.copy(expandedVals[s], d);
    }
    expandedVals = (+ scan expandedVals);
    return expandedVals;
  }

  proc broadcast(segs: [?sD] int, vals: [sD] bool, size: int) {
    var expandedVals = makeDistArray(size, uint(8));
    var diffs: [sD] uint(8);
    forall (i, d, v) in zip(sD, diffs, vals) {
      if i == sD.low {
        d = v:uint(8);
      } else {
        d = v:uint(8) - vals[i-1]:uint(8);
      }
    }
    forall (s, d) in zip(segs, diffs) with (var agg = newDstAggregator(uint(8))) {
      agg.copy(expandedVals[s], d);
    }
    expandedVals = (+ scan expandedVals);
    return (expandedVals == 1);
  }
}