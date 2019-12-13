module PerLocaleHelper {
  use CommAggregation;
  
  /*
  Takes a sorted array, computes the segments and unique keys.

  :arg sorted: sorted array
  :type sorted: [?D] int
  */
  proc segsAndUkeysFromSortedArray(sorted: [?D] int) {
    var truth: [D] bool;
    // truth array to hold segment break points
    truth[D.low] = true;
    [(t, s, i) in zip(truth, sorted, D)] if i > D.low { t = (sorted[i-1] != s); }

    // +scan to compute segment position... 1-based because of inclusive-scan
    var iv: [D] int = (+ scan truth);
    // compute how many segments
    var pop = iv[D.high];
      
    var segs: [0..#pop] int;
    var ukeys: [0..#pop] int;

    // segment position... 1-based needs to be converted to 0-based because of inclusive-scan
    // where ever a segment break is... that index is a segment start index
    forall i in D with (var agg = newDstAggregator(int)) {
      if (truth[i] == true) {
        var idx = i;
        agg.copy(segs[iv[i]-1], idx);
      }
    }
    // pull out the first key in each segment as a unique value
    [i in segs.domain] ukeys[i] = sorted[segs[i]];
    return (segs, ukeys);
  }

  proc localHistArgSort(iv:[] int, a:[?D] int, lmin: int, bins: int) {
    var hist: [0..#bins] atomic int;
    // Make counts for each value in a
    [val in a] hist[val - lmin].add(1);
    // Figure out segment offsets
    var counts = [c in hist] c.read();
    var offsets = (+ scan counts) - counts;
    // Now insert the a_index into iv 
    var binpos: [0..#bins] atomic int;
    forall (aidx, val) in zip(D, a) with (ref binpos, ref iv) {
      // Use val to determine where in iv to put a_index
      // ividx is the offset of val's bin plus a running counter
      var ividx = offsets[val - lmin] + binpos[val - lmin].fetchAdd(1);
      iv[ividx] = aidx;
    }
  }
}
