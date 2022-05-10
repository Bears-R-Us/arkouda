/* Radix Sort Least Significant Digit */
module RadixSortLSD
{
    config const RSLSD_vv = false;
    const vv = RSLSD_vv; // these need to be const for comms/performance reasons
    
    config const RSLSD_numTasks = here.maxTaskPar; // tasks per locale based on locale0
    const numTasks = RSLSD_numTasks; // tasks per locale
    const Tasks = {0..#numTasks}; // these need to be const for comms/performance reasons
    
    private param bitsPerDigit = RSLSD_bitsPerDigit; // these need to be const for comms/performance reasons
    private param numBuckets = 1 << bitsPerDigit; // these need to be const for comms/performance reasons
    

    use BlockDist;
    use BitOps;
    use AryUtil;
    use CommAggregation;
    use IO;
    use CTypes;
    use Reflection;
    use RangeChunk;
    use Logging;
    use ServerConfig;

    private config const logLevel = ServerConfig.logLevel;
    const rsLogger = new Logger(logLevel);

    record KeysComparator {
      inline proc key(k) { return k; }
    }

    record KeysRanksComparator {
      inline proc key(kr) { const (k, _) = kr; return k; }
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

    /* Radix Sort Least Significant Digit
       In-place radix sort a block distributed array
       comparator is used to extract the key from array elements
     */
    use Timers;
    enum SortPhases {
      checkSorted=0,  // check if arrays are sorted
      arrayCreate,    // create tmp and bucket arrays
      computeLCounts, // compute local counts
      scatterLCounts, // scatter local counts to global counts
      scanGCounts,    // scan global counts
      gatherLCounts,  // gather global counts to local counts
      putSorted,      // put elements in sorted location
      swapArray,      // swap main and tmp array
      rankAssign      // create result array
    };

    private config param enableTimers = false;
    var timers: enumTimers(SortPhases, enableTimers);

    private proc radixSortLSDCore(a:[?aD] ?t, nBits, negs, comparator) {
        try! rsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "type = %s nBits = %t".format(t:string,nBits));
        timers.mark(SortPhases.arrayCreate);
        var temp = a;
        
        // create a global count array to scan
        var gD = newBlockDom({0..#(numLocales * numTasks * numBuckets)});
        var globalCounts: [gD] int;
        timers.mark(SortPhases.arrayCreate);
        
        // loop over digits
        for rshift in {0..#nBits by bitsPerDigit} {
            const last = (rshift + bitsPerDigit) >= nBits;
            try! rsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "rshift = %t".format(rshift));
            // count digits
            coforall loc in Locales {
                on loc {
                    // allocate counts
                    timers.mark(SortPhases.computeLCounts, barrier=false, timingTask=loc.id==0);
                    var tasksBucketCounts: [Tasks] [0..#numBuckets] int;
                    coforall task in Tasks {
                        ref taskBucketCounts = tasksBucketCounts[task];
                        // get local domain's indices
                        var lD = aD.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
                        // count digits in this task's part of the array
                        for i in tD {
                            const key = comparator.key(temp.localAccess[i]);
                            var bucket = getDigit(key, rshift, last, negs); // calc bucket from key
                            taskBucketCounts[bucket] += 1;
                        }
                    }//coforall task
                    timers.mark(SortPhases.computeLCounts, barrier=true, timingTask=loc.id==0);

                    // write counts in to global counts in transposed order
                    timers.mark(SortPhases.scatterLCounts, barrier=false, timingTask=loc.id==0);
                    coforall tid in Tasks {
                        var aggregator = newDstAggregator(int);
                        for task in Tasks {
                            ref taskBucketCounts = tasksBucketCounts[task];
                            for bucket in chunk(0..#numBuckets, numTasks, tid) {
                                aggregator.copy(globalCounts[calcGlobalIndex(bucket, loc.id, task)],
                                                             taskBucketCounts[bucket]);
                            }
                        }
                        aggregator.flush();
                    }//coforall task
                    timers.mark(SortPhases.scatterLCounts, barrier=true, timingTask=loc.id==0);
                }//on loc
            }//coforall loc
            
            timers.mark(SortPhases.scanGCounts);
            // scan globalCounts to get bucket ends on each locale/task
            var globalStarts = + scan globalCounts;
            globalStarts -= globalCounts;
            timers.mark(SortPhases.scanGCounts);
            
            if vv {printAry("globalCounts =",globalCounts);try! stdout.flush();}
            if vv {printAry("globalStarts =",globalStarts);try! stdout.flush();}
            
            // calc new positions and permute
            coforall loc in Locales {
                on loc {
                    timers.mark(SortPhases.gatherLCounts, barrier=false, timingTask=loc.id==0);
                    // allocate counts
                    var tasksBucketPos: [Tasks] [0..#numBuckets] int;
                    // read start pos in to globalStarts back from transposed order
                    coforall tid in Tasks {
                        var aggregator = newSrcAggregator(int);
                        for task in Tasks {
                            ref taskBucketPos = tasksBucketPos[task];
                            for bucket in chunk(0..#numBuckets, numTasks, tid) {
                              aggregator.copy(taskBucketPos[bucket],
                                         globalStarts[calcGlobalIndex(bucket, loc.id, task)]);
                            }
                        }
                        aggregator.flush();
                    }//coforall task
                    timers.mark(SortPhases.gatherLCounts, barrier=true, timingTask=loc.id==0);

                    timers.mark(SortPhases.putSorted, barrier=true, timingTask=loc.id==0);
                    coforall task in Tasks {
                        ref taskBucketPos = tasksBucketPos[task];
                        // get local domain's indices
                        var lD = aD.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
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
                    timers.mark(SortPhases.putSorted, barrier=true, timingTask=loc.id==0);
                }//on loc
            }//coforall loc

            // copy back to temp for next iteration
            // Only do this if there are more digits left
            timers.mark(SortPhases.swapArray);
            if !last {
              temp <=> a;
            }
            timers.mark(SortPhases.swapArray);
        } // for rshift
    }//proc radixSortLSDCore

    proc radixSortLSD(a:[?aD] ?t, checkSorted: bool = true): [aD] (t, int) {
        var kr: [aD] (t,int) = [(key,rank) in zip(a,aD)] (key,rank);
        if (checkSorted && isSorted(a)) {
            return kr;
        }
        var (nBits, negs) = getBitWidth(a);
        radixSortLSDCore(kr, nBits, negs, new KeysRanksComparator());
        return kr;
    }

    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning a permutation vector as a block distributed array */
    proc radixSortLSD_ranks(a:[?aD] ?t, checkSorted: bool = true): [aD] int {
        timers.mark(SortPhases.checkSorted);
        if (checkSorted && isSorted(a)) {
            var ranks: [aD] int = [i in aD] i;
            timers.mark(SortPhases.checkSorted);
            return ranks;
        }
        timers.mark(SortPhases.checkSorted);

        timers.mark(SortPhases.arrayCreate);
        var kr: [aD] (t,int) = [(key,rank) in zip(a,aD)] (key,rank);
        timers.mark(SortPhases.arrayCreate);

        var (nBits, negs) = getBitWidth(a);
        radixSortLSDCore(kr, nBits, negs, new KeysRanksComparator());

        timers.mark(SortPhases.rankAssign);
        var ranks: [aD] int = [(_, rank) in kr] rank;
        timers.mark(SortPhases.rankAssign);

        writeln(timers);
        timers.clear();
        return ranks;
    }

    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning sorted keys as a block distributed array */
    proc radixSortLSD_keys(a: [?aD] ?t, checkSorted: bool = true): [aD] t {
        var copy = a;
        if (checkSorted && isSorted(a)) {
            return copy;
        }
        var (nBits, negs) = getBitWidth(a);
        radixSortLSDCore(copy, nBits, negs, new KeysComparator());
        return copy;
    }

    proc radixSortLSD_memEst(size: int, itemsize: int) {
        // 2 temp key+ranks arrays + globalStarts/globalClounts
        return (2 * size * (itemsize + numBytes(int))) +
               (2 * numLocales * numTasks * numBuckets * numBytes(int));
    }

    proc radixSortLSD_keys_memEst(size: int, itemsize: int) {
        // 2 temp key arrays + globalStarts/globalClounts
        return (2 * size * itemsize) +
               (2 * numLocales * numTasks * numBuckets * numBytes(int));

    }
}

