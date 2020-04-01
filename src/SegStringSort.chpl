module SegStringSort {
  use SegmentedArray;
  use Sort;
  use Time;
  use IO;
  use CommAggregation;
  use PrivateDist;

  private config const SSS_v = true;
  private const v = SSS_v;
  private config const SSS_numTasks = here.maxTaskPar;
  private const numTasks = SSS_numTasks;
  private config const SSS_MINBYTES = 8;
  private const MINBYTES = SSS_MINBYTES;
  private config const SSS_MEMFACTOR = 5;
  private const MEMFACTOR = SSS_MEMFACTOR;

  record StringIntComparator {
    proc keyPart(a: (string, int), i: int) {
      var len = a[0].numBytes;
      var section = if i <= len then 0:int(8) else -1:int(8);
      var part = if i <= len then a[0].byte(i) else 0:uint(8);
      return (section, part);
    }
  }
  
  proc twoPhaseStringSort(ss: SegString): [ss.offsets.aD] int throws {
    var t = getCurrentTime();
    const lengths = ss.getLengths();
    if v { writeln("Found lengths in %t seconds".format(getCurrentTime() - t)); stdout.flush(); t = getCurrentTime(); }
    // Compute length survival function and choose a pivot length
    const (pivot, nShort) = getPivot(lengths);
    if v { writeln("Computed pivot in %t seconds".format(getCurrentTime() - t)); writeln("Pivot = %t, nShort = %t".format(pivot, nShort)); stdout.flush(); t = getCurrentTime(); }
    const longStart = ss.offsets.aD.low + nShort;
    const isLong = (lengths >= pivot);
    var locs = [i in ss.offsets.aD] i;
    var longLocs = + scan isLong;
    locs -= longLocs;
    var gatherInds: [ss.offsets.aD] int;
    forall (i, l, ll, t) in zip(ss.offsets.aD, locs, longLocs, isLong) 
      with (var agg = newDstAggregator(int)) {
      if !t {
        agg.copy(gatherInds[l], i);
      } else {
        agg.copy(gatherInds[longStart+ll-1], i);
      }
    }
    if v { writeln("Partitioned short/long strings in %t seconds".format(getCurrentTime() - t)); stdout.flush(); }
    on Locales[Locales.domain.high] {
      var tl = getCurrentTime();
      const ref highDom = {longStart..ss.offsets.aD.high};
      ref highInds = gatherInds[highDom];
      // Get local copy of the long strings as Chapel strings, and their original indices
      var stringsWithInds = gatherLongStrings(ss, lengths, highInds);
      if v {writeln("Gathered long strings in %t seconds".format(getCurrentTime() - tl)); stdout.flush(); tl = getCurrentTime(); }
      // Sort the strings, but bring the inds along for the ride
      const myComparator = new StringIntComparator();
      sort(stringsWithInds, comparator=myComparator);
      if v { writeln("Sorted long strings in %t seconds".format(getCurrentTime() - tl)); stdout.flush(); tl = getCurrentTime(); }
      forall (h, s) in zip(highDom, stringsWithInds.domain) with (var agg = newDstAggregator(int)) {
        agg.copy(gatherInds[h], stringsWithInds[s][1]);
      }
      if v { writeln("Permuted long inds in %t seconds".format(getCurrentTime() - tl)); stdout.flush(); }
    }
    if v { t = getCurrentTime(); }
    const ranks = radixSortLSD_raw(ss.offsets.a, lengths, ss.values.a, gatherInds, pivot);
    if v { writeln("Sorted ranks in %t seconds".format(getCurrentTime() - t)); stdout.flush(); }
    return ranks;
  }
  
  proc getHeads(ss: SegString, lengths: [?D] int, pivot: int) {
    ref va = ss.values.a;
    var heads: [D] [0..#pivot] uint(8);
    forall (o, l, h) in zip(ss.offsets.a, lengths, heads) {
      const len = min(l, pivot);
      for j in 0..#len {
        // TODO which one is local
        use UnorderedCopy;
        unorderedCopy(h[j], va[o+j]);
      }
    }
    return heads;
  }
  
  proc getPivot(lengths: [?D] int): 2*int {
    const NBINS = 2**16;
    const BINDOM = {0..#NBINS};
    var pBins: [PrivateSpace][BINDOM] int;
    coforall loc in Locales {
      on loc {
        const lD = D.localSubdomain();
        ref locLengths = lengths.localSlice[lD];
        var locBins: [0..#numTasks][BINDOM] int;
        coforall task in 0..#numTasks {
          const tD = calcBlock(task, lD.low, lD.high);
          for i in tD {
            var bin = min(locLengths[i], NBINS-1);
            // Count number of *bytes* in bin, not the number of strings
            locBins[task][bin] += locLengths[i];
          }
        }
        pBins[here.id] = + reduce [task in 0..#numTasks] locBins[task];
      }
    }
    const bins = + reduce [loc in PrivateSpace] pBins[loc];
    // Number of bytes in strings longer than or equal to the current bin
    const tailPop = (+ reduce bins) - (+ scan bins) + bins;
    // Find the largest value of "long" such that long strings fit in one local subdomain
    const singleLocale = (tailPop < (MEMFACTOR * D.localSubdomain().size));
    var (dummy, pivot) = maxloc reduce zip(singleLocale, BINDOM);
    // Pivot should be even and not less than MINBYTES
    pivot = max(pivot + (pivot % 2), MINBYTES);
    // How many strings are "short"?
    const nShort = + reduce (lengths < pivot);
    return (pivot, nShort);
  }
  
  proc gatherLongStrings(ss: SegString, lengths: [] int, longInds: [?D] int): [] (string, int) {
    ref oa = ss.offsets.a;
    ref va = ss.values.a;
    const myD: domain(1) = D;
    const myInds: [myD] int = longInds;
    var stringsWithInds: [myD] (string, int);
    forall (i, si) in zip(myInds, stringsWithInds) {
      const l = lengths[i];
      var buf: [0..#(l+1)] uint(8);
      buf[{0..#l}] = va[{oa[i]..#l}];
      si[0] = try! createStringWithBorrowedBuffer(c_ptrTo(buf), l, l+1);
      si[1] = i;
    }
    return stringsWithInds;
  }

  // calculate sub-domain for task
  inline proc calcBlock(task: int, low: int, high: int) {
    var totalsize = high - low + 1;
    var div = totalsize / numTasks;
    var rem = totalsize % numTasks;
    var rlow: int;
    var rhigh: int;
    if (task < rem) {
      rlow = task * (div+1) + low;
      rhigh = rlow + div;
    }
    else {
      rlow = task * div + rem + low;
      rhigh = rlow + div - 1;
    }
    return {rlow .. rhigh};
  }

  // calc global transposed index
  // (bucket,loc,task) = (bucket * numLocales * numTasks) + (loc * numTasks) + task;
  inline proc calcGlobalIndex(bucket: int, loc: int, task: int): int {
    return ((bucket * numLocales * numTasks) + (loc * numTasks) + task);
  }
  
  proc radixSortLSD_raw(const ref offsets: [?aD] int, const ref lengths: [aD] int, const ref values: [] uint(8), const ref inds: [aD] int, const pivot: int): [aD] int throws {
    const numBuckets = 2**16;
    type state = (uint(8), uint(8), int, int, int);
    inline proc copyDigit(ref k: state, const off: int, const len: int, const rank: int, const right: int) {
      // TODO can we only use the aggregated version?
      use UnorderedCopy;
      if (right > 0) {
        if (len >= right) {
          unorderedCopy(k[0], values[off+right-2]);
          unorderedCopy(k[1], values[off+right-1]);
        } else if (len == right - 1) {
          unorderedCopy(k[0], values[off+right-2]);
          unorderedCopy(k[1], 0: uint(8));
        } else {
          unorderedCopy(k[0], 0: uint(8));
          unorderedCopy(k[1], 0: uint(8));
        }
      }
      unorderedCopy(k[2], off);
      unorderedCopy(k[3], len);
      unorderedCopy(k[4], rank);
    }

    inline proc copyDigit(ref k1: state, ref k0: state, const right: int, ref aggregator) {
      const off = k0[2];
      const len = k0[3];
      if (right > 0) {
        if (len >= right) {
          k0[0] = values[off+right-2];
          k0[1] = values[off+right-1];
        } else if (len == right - 1) {
          k0[0] = values[off+right-2];
          k0[1] = 0;
        } else {
          k0[0] = 0: uint(8);
          k0[1] = 0: uint(8);
        }
      }
      aggregator.copy(k1, k0);
    }
    
    var kr0: [aD] state;
    if v { writeln("rshift = 0"); stdout.flush(); }
    forall (k, rank) in zip(kr0, inds) {
      copyDigit(k, offsets[rank], lengths[rank], rank, pivot);
    }
    var kr1: [aD] state;
    // create a global count array to scan
    var gD = newBlockDom({0..#(numLocales * numTasks * numBuckets)});
    var globalCounts: [gD] int;
    var globalStarts: [gD] int;
        
    // loop over digits
    for rshift in {2..#pivot by 2} {
      if v {writeln("rshift = ",rshift); stdout.flush();}
      // count digits
      coforall loc in Locales {
        on loc {
          coforall task in 0..#numTasks {
            // bucket domain
            var bD = {0..#numBuckets};
            // allocate counts
            var taskBucketCounts: [bD] int;
            // get local domain's indices
            var lD = kr0.localSubdomain();
            // calc task's indices from local domain's indices
            var tD = calcBlock(task, lD.low, lD.high);
            // count digits in this task's part of the array
            for i in tD {
              var bucket = (kr0[i][0]:int << 8) | (kr0[i][1]:int); // calc bucket from key
              taskBucketCounts[bucket] += 1;
            }
            // write counts in to global counts in transposed order
            var aggregator = newDstAggregator(int);
            for bucket in bD {
              aggregator.copy(globalCounts[calcGlobalIndex(bucket, loc.id, task)], taskBucketCounts[bucket]);
            }
            aggregator.flush();
          }//coforall task
        }//on loc
      }//coforall loc
            
      // scan globalCounts to get bucket ends on each locale/task
      globalStarts = + scan globalCounts;
      globalStarts = globalStarts - globalCounts;
            
      // calc new positions and permute
      coforall loc in Locales {
        on loc {
          coforall task in 0..#numTasks {
            // bucket domain
            var bD = {0..#numBuckets};
            // allocate counts
            var taskBucketPos: [bD] int;
            // get local domain's indices
            var lD = kr0.localSubdomain();
            // calc task's indices from local domain's indices
            var tD = calcBlock(task, lD.low, lD.high);
            // read start pos in to globalStarts back from transposed order
            {
              var aggregator = newSrcAggregator(int);
              for bucket in bD {
                aggregator.copy(taskBucketPos[bucket], globalStarts[calcGlobalIndex(bucket, loc.id, task)]);
              }
              aggregator.flush();
            }
            // calc new position and put (key,rank) pair there in kr1
            {
              var aggregator = newDstAggregator(state);
              for i in tD {
                var bucket = (kr0[i][0]:int << 8) | (kr0[i][1]:int); // calc bucket from key
                var pos = taskBucketPos[bucket];
                taskBucketPos[bucket] += 1;
                copyDigit(kr1[pos], kr0[i], pivot - rshift, aggregator);
              }
              aggregator.flush();
            }
          }//coforall task 
        }//on loc
      }//coforall loc
            
      // copy back to k0 and r0 for next iteration
      // Only do this if there are more digits left
      // If this is the last digit, the negative-swapping code will copy the ranks
      if (rshift < pivot) {
        kr0 = kr1;
      }
    } // for rshift
    const ranks: [aD] int = [(a, b, c, d, i) in kr1] i;
    return ranks;
  }
}
