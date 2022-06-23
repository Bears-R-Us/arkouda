/* Radix Sort Least Significant Digit */
module RadixSortLSD {
    use BlockDist;
    use BitOps;
    use AryUtil;
    use CommAggregation;
    use IO;
    use CTypes;
    use Reflection;
    use Logging;
    use ServerConfig;

    private config const logLevel = ServerConfig.logLevel;
    const rsLogger = new Logger(logLevel);

    record RadixSortLSDPlan {
        const numTasks: int; // tasks per locale
        const Tasks: domain;
        const bitsPerDigit: int;
        const numBuckets;

        proc init(numTasks: int = here.maxTaskPar, bitsPerDigit: int = RSLSD_bitsPerDigit) {
            this.numTasks = numTasks;
            this.Tasks = {0..#numTasks};
            this.bitsPerDigit = bitsPerDigit;
            this.numBuckets = 1 << bitsPerDigit;
        }
    }

    record KeysComparator {
      inline proc key(k) { return k; }
    }

    record KeysRanksComparator {
      inline proc key(kr) { const (k, _) = kr; return k; }
    }

    // calculate sub-domain for task
    inline proc calcBlock(task: int, low: int, high: int, numTasks: int) {
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
    inline proc calcGlobalIndex(bucket: int, loc: int, task: int, numTasks: int): int {
        return ((bucket * numLocales * numTasks) + (loc * numTasks) + task);
    }

    /* Radix Sort Least Significant Digit
       In-place radix sort a block distributed array
       comparator is used to extract the key from array elements
     */
    private proc radixSortLSDCore(a:[?aD] ?t, nBits, negs, comparator, const plan: RadixSortLSDPlan) {
        try! rsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "type = %s nBits = %t".format(t:string,nBits));
        var temp = a;
        
        // create a global count array to scan
        var gD = newBlockDom({0..#(numLocales * plan.numTasks * plan.numBuckets)});
        var globalCounts: [gD] int;
        
        // loop over digits
        for rshift in {0..#nBits by plan.bitsPerDigit} {
            const last = (rshift + plan.bitsPerDigit) >= nBits;
            try! rsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "rshift = %t".format(rshift));
            // count digits
            coforall loc in Locales {
                on loc {
                    coforall task in plan.Tasks {
                        // bucket domain
                        var bD = {0..#plan.numBuckets};
                        // allocate counts
                        var taskBucketCounts: [bD] int;
                        // get local domain's indices
                        var lD = aD.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high, plan.numTasks);
                        try! rsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                   "locid: %t task: %t tD: %t".format(loc.id,task,tD));
                        // count digits in this task's part of the array
                        for i in tD {
                            const key = comparator.key(temp.localAccess[i]);
                            var bucket = getDigit(key, rshift, last, negs); // calc bucket from key
                            taskBucketCounts[bucket] += 1;
                        }
                        // write counts in to global counts in transposed order
                        var aggregator = newDstAggregator(int);
                        for bucket in bD {
                            aggregator.copy(globalCounts[calcGlobalIndex(bucket, loc.id, task, plan.numTasks)],
                                                         taskBucketCounts[bucket]);
                        }
                        aggregator.flush();
                    }//coforall task
                }//on loc
            }//coforall loc
            
            // scan globalCounts to get bucket ends on each locale/task
            var globalStarts = + scan globalCounts;
            globalStarts -= globalCounts;

            try! rsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "globalCounts = %t".format(globalCounts));
            try! rsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "globalStarts = %t".format(globalStarts));
            
            // calc new positions and permute
            coforall loc in Locales {
                on loc {
                    coforall task in plan.Tasks {
                        // bucket domain
                        var bD = {0..#plan.numBuckets};
                        // allocate counts
                        var taskBucketPos: [bD] int;
                        // get local domain's indices
                        var lD = aD.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high, plan.numTasks);
                        // read start pos in to globalStarts back from transposed order
                        {
                            var aggregator = newSrcAggregator(int);
                            for bucket in bD {
                                aggregator.copy(taskBucketPos[bucket], 
                                           globalStarts[calcGlobalIndex(bucket, loc.id, task, plan.numTasks)]);
                            }
                            aggregator.flush();
                        }
                        // calc new position and put data there in temp
                        {
                            var aggregator = newDstAggregator(t);
                            for i in tD {
                                const ref tempi = temp.localAccess[i];
                                const key = comparator.key(tempi);
                                var bucket = getDigit(key, rshift, last, negs); // calc bucket from key
                                var pos = taskBucketPos[bucket];
                                taskBucketPos[bucket] += 1;
                                aggregator.copy(a[pos], tempi);
                            }
                            aggregator.flush();
                        }
                    }//coforall task 
                }//on loc
            }//coforall loc

            // copy back to temp for next iteration
            // Only do this if there are more digits left
            if !last {
              temp <=> a;
            }
        } // for rshift
    }//proc radixSortLSDCore

    proc radixSortLSD(a:[?aD] ?t, checkSorted: bool = true, numTasks: int = here.maxTaskPar, bitsPerDigit: int = RSLSD_bitsPerDigit): [aD] (t, int) {
        var kr: [aD] (t,int) = [(key,rank) in zip(a,aD)] (key,rank);
        if (checkSorted && isSorted(a)) {
            return kr;
        }
        var (nBits, negs) = getBitWidth(a);
        var plan = new RadixSortLSDPlan(numTasks, bitsPerDigit);
        radixSortLSDCore(kr, nBits, negs, new KeysRanksComparator(), plan);
        return kr;
    }

    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning a permutation vector as a block distributed array */
    proc radixSortLSD_ranks(a:[?aD] ?t, checkSorted: bool = true, numTasks: int = here.maxTaskPar, bitsPerDigit: int = RSLSD_bitsPerDigit): [aD] int {
        if (checkSorted && isSorted(a)) {
            var ranks: [aD] int = [i in aD] i;
            return ranks;
        }

        var kr: [aD] (t,int) = [(key,rank) in zip(a,aD)] (key,rank);
        var (nBits, negs) = getBitWidth(a);
        var plan = new RadixSortLSDPlan(numTasks, bitsPerDigit);
        radixSortLSDCore(kr, nBits, negs, new KeysRanksComparator(), plan);
        var ranks: [aD] int = [(_, rank) in kr] rank;
        return ranks;
    }

    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning sorted keys as a block distributed array */
    proc radixSortLSD_keys(a: [?aD] ?t, checkSorted: bool = true, numTasks: int = here.maxTaskPar, bitsPerDigit: int = RSLSD_bitsPerDigit): [aD] t {
        var copy = a;
        if (checkSorted && isSorted(a)) {
            return copy;
        }
        var (nBits, negs) = getBitWidth(a);
        var plan = new RadixSortLSDPlan(numTasks, bitsPerDigit);
        radixSortLSDCore(copy, nBits, negs, new KeysComparator(), plan);
        return copy;
    }

    proc radixSortLSD_memEst(size: int, itemsize: int, numTasks: int = here.maxTaskPar, numBuckets: int = 1 << RSLSD_bitsPerDigit) {
        // 2 temp key+ranks arrays + globalStarts/globalClounts
        return (2 * size * (itemsize + numBytes(int))) +
               (2 * numLocales * numTasks * numBuckets * numBytes(int));
    }

    proc radixSortLSD_keys_memEst(size: int, itemsize: int, numTasks: int = here.maxTaskPar, numBuckets: int = 1 << RSLSD_bitsPerDigit) {
        // 2 temp key arrays + globalStarts/globalClounts
        return (2 * size * itemsize) +
               (2 * numLocales * numTasks * numBuckets * numBytes(int));

    }
}

