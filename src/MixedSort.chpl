module MixedSort {
  private use RadixSortLSD;
  private use IO;
  private use BlockDist;
  private use Reflection;
  private use CommAggregation;
  // private use AryUtil;

  private config param MSD_bitsPerDigit = 8;
  private config param LSD_bitsPerDigit = 16;
  private config const numTasks = here.maxTaskPar;

  proc mixedSort_ranks(a:[?aD] ?t, checkSorted: bool = true): [aD] int throws {
    var (nBits, hasNegatives) = getBitWidth(a);
    var kr0: [aD] (t, int) = [(key, rank) in zip(a, aD)] (key, rank);
    sortBucket(kr0, t, aD, nBits, hasNegatives, checkSorted, numTasks);
    var ranks: [aD] int = [(key, rank) in kr0] rank;
    return ranks;
  }

  proc sortBucket(kr0: [], type t, bD, curBit, hasNegatives, checkSorted, nTasks) throws {
    if bD.targetLocales().size == 1 {
      localSort(kr0, t, bD, curBit, hasNegatives, true, nTasks);
    }
    // If bucket spans multiple locales, sort the next most significant digit
    const rshift = curBit - MSD_bitsPerDigit;
    var segments = sortDigit(kr0, t, bD, rshift, hasNegatives, true, nTasks, MSD_bitsPerDigit);
    // Recurse on each digit's bucket
    for (bs, be) in segments {
      begin {
        const bDs: subdomain(bD) = bD[bs..be];
        // Give bucket it's proportion of the task pool
        const myTasks = max((nTasks * (be - bs + 1) + bD.size - 1) / bD.size, 1):int;
        // If bucket spans multiple locales, divide tasks in half
        const subTasks = if (bDs.targetLocales().size == 1) then myTasks else max(myTasks/2, 1);
        // Sort bucket and assign result to parent array
        sortBucket(kr0, t, bDs, rshift, hasNegatives, true, subTasks);
      }
    }
  }

  proc keysRanksSorted(kr:[] ?t, aD) {
    var sorted: bool = true;
    forall i in aD with (&& reduce sorted) {
      if i > aD.low {
        const (k1,_) = kr[i];
        const (k0,_) = kr[i-1];
        sorted &&= (k0 <= k1);
      }
    }
    return sorted;
  }

  proc localSort(kr0:[], type t, bD, curBit, hasNegatives, checkSorted, nTasks) throws {
    if (checkSorted) {      
      if (keysRanksSorted(kr0, bD)) {
        return;
      }
    }
    for rshift in 0..#curBit by LSD_bitsPerDigit {
      sortDigit(kr0, t, bD, rshift, hasNegatives, false, nTasks, LSD_bitsPerDigit);
    }
  }

  // calc global transposed index
  // (bucket,loc,task) = (bucket * numLocales * numTasks) + (loc * numTasks) + task;
  inline proc calcGlobalIndex(bucket: int, loc: int, task: int, nloc: int, locmin: int): int {
    return ((bucket * nloc * numTasks) + ((loc - locmin) * numTasks) + task);
  }

  private proc sortDigit(kr0:[], type t, aD, rshift: int, hasNegatives: bool, checkSorted: bool = true, nTasks: int = numTasks, param bitsPerDigit) throws {
    if (checkSorted) {
      if (keysRanksSorted(kr0, aD)) {
        const emptyBuckets: [0..#0] (int, int);
        return emptyBuckets;
      }
    }
    param numBuckets = 1 << bitsPerDigit;
    // form (key,rank) vector
    /* var kr0: [aD] (t,int) = [(key,rank) in zip(a,aD)] (key,rank); */
    var kr1: [aD] (t,int);
    const nloc = aD.targetLocales().size;
    const firstLocale = aD.targetLocales()[aD.targetLocales().domain.low];
    const lastLocale = aD.targetLocales()[aD.targetLocales().domain.high];
    const locmin = firstLocale.id;
    const last = rshift <= bitsPerDigit;
    const gDloc = {0..#(aD.targetLocales().size * nTasks * numBuckets)};
    const gD: domain(1) dmapped Block(boundingBox=gDloc, targetLocales=aD.targetLocales()) = gDloc;
    var globalCounts: [gD] int;
    var globalStarts: [gD] int;

    // count digits
    coforall loc in aD.targetLocales() {
      on loc {
        // All middle locales are fully committed to sorting this bucket and should get max tasks available
        // But first and last locales need to share, according to nTasks
        const taskPoolSize = if ((loc == firstLocale) || (loc == lastLocale)) then nTasks else numTasks;
        coforall task in 0..#taskPoolSize {
          // bucket domain
          var bD = {0..#numBuckets};
          // allocate counts
          var taskBucketCounts: [bD] int;
          // get local domain's indices
          var lD = aD.localSubdomain();
          // calc task's indices from local domain's indices
          var tD = calcBlock(task, lD.low, lD.high);
          try! rsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "locid: %t task: %t tD: %t".format(loc.id,task,tD));
          // count digits in this task's part of the array
          for i in tD {
            const (key,_) = kr0[i];
            var bucket = getDigit(key, rshift, last, hasNegatives); // calc bucket from key
            taskBucketCounts[bucket] += 1;
          }
          // write counts in to global counts in transposed order
          var aggregator = newDstAggregator(int);
          for bucket in bD {
            aggregator.copy(globalCounts[calcGlobalIndex(bucket, loc.id, task, nloc, locmin)], 
                            taskBucketCounts[bucket]);
          }
          aggregator.flush();
        }//coforall task
      }//on loc
    }//coforall loc
            
    // scan globalCounts to get bucket ends on each locale/task
    // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
    overMemLimit(numBytes(int) * globalCounts.size);
    globalStarts = + scan globalCounts;
    globalStarts = globalStarts - globalCounts + aD.low;
    var bucketRanges: [{0..#numBuckets}] (int, int);
    for bi in 0..#numBuckets {
      const gi = calcGlobalIndex(bi, locmin, 0, nloc, locmin);
      const start = globalStarts[gi];
      if (bi == numBuckets - 1) {
        bucketRanges[bi] = (start, aD.high);
      } else {
        bucketRanges[bi] = (start, globalStarts[gi+1] - 1);
      }
    }
            
    if vv {printAry("globalCounts =",globalCounts);try! stdout.flush();}
    if vv {printAry("globalStarts =",globalStarts);try! stdout.flush();}
            
    // calc new positions and permute
    coforall loc in aD.targetLocales() {
      on loc {
        // All middle locales are fully committed to sorting this bucket and should get max tasks available
        // But first and last locales need to share, according to nTasks
        const taskPoolSize = if ((loc == firstLocale) || (loc == lastLocale)) then nTasks else numTasks;
        coforall task in 0..#taskPoolSize {
          // bucket domain
          var bD = {0..#numBuckets};
          // allocate counts
          var taskBucketPos: [bD] int;
          // get local domain's indices
          var lD = aD.localSubdomain();
          // calc task's indices from local domain's indices
          var tD = calcBlock(task, lD.low, lD.high);
          // read start pos in to globalStarts back from transposed order
          {
            var aggregator = newSrcAggregator(int);
            for bucket in bD {
              aggregator.copy(taskBucketPos[bucket], 
                              globalStarts[calcGlobalIndex(bucket, loc.id, task, nloc, locmin)]);
            }
            aggregator.flush();
          }
          // calc new position and put (key,rank) pair there in kr1
          {
            var aggregator = newDstAggregator((t,int));
            for i in tD {
              const (key,_) = kr0[i];
              var bucket = getDigit(key, rshift, last, hasNegatives); // calc bucket from key
              var pos = taskBucketPos[bucket];
              taskBucketPos[bucket] += 1;
              aggregator.copy(kr1[pos], kr0[i]);
            }
            aggregator.flush();
          }
        }//coforall task 
      }//on loc
    }//coforall loc

    forall i in aD {
      kr0[i] = kr1[i];
    }
    return bucketRanges;
  }
}