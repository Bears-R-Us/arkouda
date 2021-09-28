module MixedSort {
  private use IO;
  private use BitOps;
  private use BlockDist;
  private use Reflection;
  private use CommAggregation;
  private use Logging;
  private use ServerConfig;
  // private use AryUtil;

  private config param MSD_bitsPerDigit = 8;
  private config param LSD_bitsPerDigit = 16;
  private config const numTasks = here.maxTaskPar;
  private config const logLevel = ServerConfig.logLevel;
  const msLogger = new Logger(logLevel);

  inline proc getBitWidth(a: [?aD] int): (int, bool) {
    var aMin = min reduce a;
    var aMax = max reduce a;
    var wPos = if aMax >= 0 then numBits(int) - clz(aMax) else 0;
    var wNeg = if aMin < 0 then numBits(int) - clz((-aMin)-1) + 1 else 0;
    const bitWidth = max(wPos, wNeg);
    const negs = aMin < 0;
    return (bitWidth, negs);
  }
  
  proc mixedSort_ranks(a:[?aD] ?t, checkSorted: bool = true): [aD] int throws {
    var (nBits, hasNegatives) = getBitWidth(a);
    msLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                   "nBits = %t, hasNegatives = %t".format(nBits, hasNegatives));
    var kr0: [aD] (t, int) = [(key, rank) in zip(a, aD)] (key, rank);
    sortBucket(kr0, t, aD, nBits, hasNegatives, checkSorted, numTasks);
    var ranks: [aD] int = [(key, rank) in kr0] rank;
    return ranks;
  }

  proc sortBucket(kr0: [], type t, bD, curBit, hasNegatives, checkSorted, nTasks) throws {
    if bD.size == 0 {
      return;
    }
    ref first = kr0[bD.low];
    ref last = kr0[bD.high];
    if first.locale.id == last.locale.id {
      localSort(kr0, t, bD, curBit, hasNegatives, true, nTasks);
      return;
    }
    // If bucket spans multiple locales, sort the next most significant digit
    const rshift = max(0, curBit - MSD_bitsPerDigit);
    // var kr0: [aD] (t, int) = [(key, rank) in zip(a, aD)] (key, rank);
    var segments = sortDigit(kr0, t, bD, rshift, hasNegatives, true, nTasks, MSD_bitsPerDigit);
    if rshift == 0 {
      return;
    }
    // Recurse on each digit's bucket
    sync for (bs, be) in segments {
      if (be >= bs) {
        begin {
          const bDs: subdomain(bD) = bD[bs..be];
          // Give bucket it's proportion of the task pool
          const myTasks = max((nTasks * (be - bs + 1) + bD.size - 1) / bD.size, 1):int;
          // If bucket spans multiple locales, divide tasks in half
          ref myfirst = kr0[bs];
          ref mylast = kr0[be];
          const subTasks = if (myfirst.locale.id == mylast.locale.id) then myTasks else max(myTasks/2, 1);
          // Sort bucket and assign result to parent array
          sortBucket(kr0, t, bDs, rshift, hasNegatives, true, subTasks);
        }
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
    if bD.size == 0 {
      return;
    }
    if (checkSorted) {      
      if (keysRanksSorted(kr0, bD)) {
        return;
      }
    }
    for rshift in 0..#curBit by LSD_bitsPerDigit {
      sortDigit(kr0, t, bD, rshift, hasNegatives, false, nTasks, LSD_bitsPerDigit);
    }
  }

    // Get the digit for the current rshift. In order to correctly sort
    // negatives, we have to invert the signbit if we're looking at the last
    // digit and the array contained negative values.
    inline proc getDigit(key: int, rshift: int, last: bool, negs: bool, param bitsPerDigit): int {
      param maskDigit = (1 << bitsPerDigit) - 1;
      const invertSignBit = last && negs;
      const xor = (invertSignBit:uint << (bitsPerDigit-1));
      const keyu = key:uint;
      return (((keyu >> rshift) & (maskDigit:uint)) ^ xor):int;
    }

    inline proc getDigit(key: uint, rshift: int, last: bool, negs: bool, param bitsPerDigit): int {
      param maskDigit = (1 << bitsPerDigit) - 1;
      return ((key >> rshift) & (maskDigit:uint)):int;
    }

    // Get the digit for the current rshift. In order to correctly sort
    // negatives, we have to invert the entire key if it's negative, and invert
    // just the signbit for positive values when looking at the last digit.
    inline proc getDigit(in key: real, rshift: int, last: bool, negs: bool, param bitsPerDigit): int {
      param maskDigit = (1 << bitsPerDigit) - 1;
      const invertSignBit = last && negs;
      var keyu: uint;
      c_memcpy(c_ptrTo(keyu), c_ptrTo(key), numBytes(key.type));
      var signbitSet = keyu >> (numBits(keyu.type)-1) == 1;
      var xor = 0:uint;
      if signbitSet {
        keyu = ~keyu;
      } else {
        xor = (invertSignBit:uint << (bitsPerDigit-1));
      }
      return (((keyu >> rshift) & (maskDigit:uint)) ^ xor):int;
    }

    inline proc getDigit(key: 2*uint, rshift: int, last: bool, negs: bool, param bitsPerDigit): int {
      const (key0,key1) = key;
      if (rshift >= numBits(uint)) {
        return getDigit(key0, rshift - numBits(uint), last, negs, bitsPerDigit);
      } else {
        return getDigit(key1, rshift, last, negs, bitsPerDigit);
      }
    }

    inline proc getDigit(key: _tuple, rshift: int, last: bool, negs: bool, param bitsPerDigit): int
        where isHomogeneousTuple(key) && key.type == key.size*uint(bitsPerDigit) {
      const keyHigh = key.size - 1;
      return key[keyHigh - rshift/bitsPerDigit]:int;
    }

    // calculate sub-domain for task
    inline proc calcBlock(task: int, low: int, high: int, numTasksHere: int) {
        var totalsize = high - low + 1;
        var div = totalsize / numTasksHere;
        var rem = totalsize % numTasksHere;
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
  inline proc calcGlobalIndex(bucket: int, loc: int, task: int, nloc: int, locmin: int): int {
    return ((bucket * nloc * numTasks) + ((loc - locmin) * numTasks) + task);
  }

  private proc sortDigit(kr0:[], type t, aD, rshift: int, hasNegatives: bool, checkSorted: bool = true, nTasks: int = numTasks, param bitsPerDigit) throws {
    ref first = kr0[aD.low];
    ref last = kr0[aD.high];
    const firstLocale = first.locale.id;
    const lastLocale = last.locale.id;
    const myLocales = Locales[firstLocale..lastLocale];
    const nloc = myLocales.size;
    msLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                   "rshift = %t, aD = %t, loc = %t, nTasks = %t, bitsPerDigit = %t)".format(rshift, aD, myLocales, nTasks, bitsPerDigit));
    const emptyBuckets: [0..#0] (int, int);
    if aD.size == 0 {
      return emptyBuckets;
    }
    if (checkSorted) {
      if (keysRanksSorted(kr0, aD)) {
        return emptyBuckets;
      }
    }
    param numBuckets = 1 << bitsPerDigit;
    // form (key,rank) vector
    /* var kr0: [aD] (t,int) = [(key,rank) in zip(a,aD)] (key,rank); */
    var kr1: [aD] (t,int);
    const isLast = rshift <= bitsPerDigit;
    // Make buckets for all cores on all locales, even if some aren't used
    const gDloc = {0..#(nloc * numTasks * numBuckets)};
    const gD: domain(1) dmapped Block(boundingBox=gDloc, targetLocales=myLocales) = gDloc;
    var globalCounts: [gD] int;
    var globalStarts: [gD] int;

    // count digits
    coforall loc in myLocales {
      on loc {
        // All middle locales are fully committed to sorting this bucket and should get max tasks available
        // But first and last locales need to share, according to nTasks
        const taskPoolSize = if ((loc.id == firstLocale) || (loc.id == lastLocale)) then nTasks else numTasks;
        coforall task in 0..#taskPoolSize {
          // bucket domain
          var bD = {0..#numBuckets};
          // allocate counts
          var taskBucketCounts: [bD] int;
          // get local domain's indices
          var lD = aD.localSubdomain();
          // calc task's indices from local domain's indices
          var tD = calcBlock(task, lD.low, lD.high, taskPoolSize);
          msLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "locid: %t task: %t tD: %t lD: %t globalCountDom: %t".format(loc.id,task,tD,lD, {calcGlobalIndex(0, loc.id, task, nloc, firstLocale)..calcGlobalIndex(numBuckets-1, loc.id, task, nloc, firstLocale)}));
          // count digits in this task's part of the array
          for i in tD {
            const (key,_) = kr0[i];
            var bucket = getDigit(key, rshift, isLast, hasNegatives, bitsPerDigit); // calc bucket from key
            taskBucketCounts[bucket] += 1;
          }
          // write counts in to global counts in transposed order
          var aggregator = newDstAggregator(int);
          for bucket in bD {
            aggregator.copy(globalCounts[calcGlobalIndex(bucket, loc.id, task, nloc, firstLocale)], 
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
      // Bucket start is global index of bucket for min locale and task
      const gistart = calcGlobalIndex(bi, firstLocale, 0, nloc, firstLocale);
      const start = globalStarts[gistart];
      if (bi == numBuckets - 1) {
        // End of last bucket is end of domain
        bucketRanges[bi] = (start, aD.high);
      } else {
        // Bucket end: find global start idx of next bucket and decrement
        const giend = calcGlobalIndex(bi+1, firstLocale, 0, nloc, firstLocale);
        bucketRanges[bi] = (start, globalStarts[giend] - 1);
      }
    }
            
    // if vv {printAry("globalCounts =",globalCounts);try! stdout.flush();}
    // if vv {printAry("globalStarts =",globalStarts);try! stdout.flush();}
            
    // calc new positions and permute
    coforall loc in myLocales {
      on loc {
        // All middle locales are fully committed to sorting this bucket and should get max tasks available
        // But first and last locales need to share, according to nTasks
        const taskPoolSize = if ((loc.id == firstLocale) || (loc.id == lastLocale)) then nTasks else numTasks;
        coforall task in 0..#taskPoolSize {
          // bucket domain
          var bD = {0..#numBuckets};
          // allocate counts
          var taskBucketPos: [bD] int;
          // get local domain's indices
          var lD = aD.localSubdomain();
          // calc task's indices from local domain's indices
          var tD = calcBlock(task, lD.low, lD.high, taskPoolSize);
          // read start pos in to globalStarts back from transposed order
          {
            var aggregator = newSrcAggregator(int);
            for bucket in bD {
              aggregator.copy(taskBucketPos[bucket], 
                              globalStarts[calcGlobalIndex(bucket, loc.id, task, nloc, firstLocale)]);
            }
            aggregator.flush();
          }
          // calc new position and put (key,rank) pair there in kr1
          {
            var aggregator = newDstAggregator((t,int));
            for i in tD {
              const (key,_) = kr0[i];
              var bucket = getDigit(key, rshift, isLast, hasNegatives, bitsPerDigit); // calc bucket from key
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
