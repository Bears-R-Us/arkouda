module SegStringSort {
  use SegmentedArray;
  use Sort;
  use Time;
  use IO;
  use CPtr;
  use CommAggregation;
  use PrivateDist;
  use Reflection;
  use Logging;
  use ServerConfig;

  private config const SSS_v = false;
  private const vv = SSS_v;
  private config const SSS_numTasks = here.maxTaskPar;
  private const numTasks = SSS_numTasks;
  private config const SSS_MINBYTES = 8;
  private const MINBYTES = SSS_MINBYTES;
  private config const SSS_MEMFACTOR = 5;
  private const MEMFACTOR = SSS_MEMFACTOR;
  private config const SSS_PARTITION_LONG_STRING = false;
  private const PARTITION_LONG_STRING = SSS_PARTITION_LONG_STRING;
 
  
  const ssLogger = new Logger();
  if v {
      ssLogger.level = LogLevel.DEBUG;
  } else {
      ssLogger.level = LogLevel.INFO;    
  }

  record StringIntComparator {
    proc keyPart((a0,_): (string, int), in i: int) {
      // Just run the default comparator on the string
      return Sort.defaultComparator.keyPart(a0, i);
    }
  }
  
  proc twoPhaseStringSort(ss: SegString): [ss.offsets.aD] int throws {
    var t = getCurrentTime();
    const lengths = ss.getLengths();
    ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "Found lengths in %t seconds".format(getCurrentTime() - t));
    t = getCurrentTime();
    // Compute length survival function and choose a pivot length
    const (pivot, nShort) = getPivot(lengths);
    ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "Computed pivot in %t seconds".format(getCurrentTime() - t)); 
    ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "Pivot = %t, nShort = %t".format(pivot, nShort)); 
    t = getCurrentTime();
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
    ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                   "Partitioned short/long strings in %t seconds".format(getCurrentTime() - t));
    on Locales[Locales.domain.high] {
      var tl = getCurrentTime();
      const ref highDom = {longStart..ss.offsets.aD.high};
      ref highInds = gatherInds[highDom];
      // Get local copy of the long strings as Chapel strings, and their original indices
      var stringsWithInds = gatherLongStrings(ss, lengths, highInds);

      ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                           "Gathered long strings in %t seconds".format(getCurrentTime() - tl));
      tl = getCurrentTime();
      // Sort the strings, but bring the inds along for the ride
      const myComparator = new StringIntComparator();
      sort(stringsWithInds, comparator=myComparator);

      ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                             "Sorted long strings in %t seconds".format(getCurrentTime() - tl));
      tl = getCurrentTime();

      forall (h, s) in zip(highDom, stringsWithInds.domain) with (var agg = newDstAggregator(int)) {
        const (_,val) = stringsWithInds[s];
        agg.copy(gatherInds[h], val);
      }
      ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "Permuted long inds in %t seconds".format(getCurrentTime() - tl));
    }
    t = getCurrentTime();
    const ranks = radixSortLSD_raw(ss.offsets.a, lengths, ss.values.a, gatherInds, pivot);
    ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                          "Sorted ranks in %t seconds".format(getCurrentTime() - t));
    return ranks;
  }
  
  proc getPivot(lengths: [?D] int): 2*int {
    if !PARTITION_LONG_STRING {
      var pivot = max reduce lengths + 1;
      pivot = max(pivot + (pivot % 2), MINBYTES);
      const nShort = D.size;
      return (pivot, nShort);
    } else {
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
      si = (try! createStringWithBorrowedBuffer(c_ptrTo(buf), l, l+1), i);
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
    inline proc copyDigit(ref k: state, const off: int, const len: int, const rank: int, const right: int, ref agg) {
      ref (k0,k1,k2,k3,k4) = k;
      if (right > 0) {
        if (len >= right) {
          agg.copy(k0, values[off+right-2]);
          agg.copy(k1, values[off+right-1]);
        } else if (len == right - 1) {
          agg.copy(k0, values[off+right-2]);
          k1 = 0: uint(8);
        } else {
          k0 = 0: uint(8);
          k1 = 0: uint(8);
        }
      }
      k2 = off;
      k3 = len;
      k4 = rank;
    }

    inline proc copyDigit(ref k0: state, const right: int, ref agg) {
      const (_,_,off,len,_) = k0;
      ref (ka,kb,_,_,_) = k0;
      if (right > 0) {
        if (len >= right) {
          agg.copy(ka, values[off+right-2]);
          agg.copy(kb, values[off+right-1]);
        } else if (len == right - 1) {
          agg.copy(ka, values[off+right-2]);
          kb = 0;
        } else {
          ka = 0: uint(8);
          kb = 0: uint(8);
        }
      }
    }
    
    var kr0: [aD] state;
    ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"rshift = 0");
    forall (k, rank) in zip(kr0, inds) with (var agg = newSrcAggregator(uint(8))) {
      copyDigit(k, offsets[rank], lengths[rank], rank, pivot, agg);
    }
    var kr1: [aD] state;
    // create a global count array to scan
    var gD = newBlockDom({0..#(numLocales * numTasks * numBuckets)});
    var globalCounts: [gD] int;
    var globalStarts: [gD] int;
        
    // loop over digits
    for rshift in {2..#pivot by 2} {
      ssLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"rshift = %t".format(rshift));
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
              var kr0i0, kr0i1: int;
              (kr0i0, kr0i1 , _, _, _) = kr0[i];
              var bucket = (kr0i0 << 8) | (kr0i1); // calc bucket from key
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
            // calculate the new position from old values/digits
            var buckets: [tD] int = for i in tD do (kr0[i][0]:int << 8) | (kr0[i][1]:int);
            // copy in current values/digits
            {
              var aggregator = newSrcAggregator(uint(8));
              for i in tD {
                copyDigit(kr0[i], pivot - rshift, aggregator);
              }
              aggregator.flush();
            }
            // put (key,rank) pair into new position in kr1
            {
              var aggregator = newDstAggregator(state);
              for i in tD {
                var bucket = buckets[i];
                var pos = taskBucketPos[bucket];
                taskBucketPos[bucket] += 1;
                aggregator.copy(kr1[pos], kr0[i]);
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
