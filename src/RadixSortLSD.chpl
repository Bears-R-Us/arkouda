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
    use Sort except isSorted;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const rsLogger = new Logger(logLevel, logChannel);

    record KeysComparator: keyComparator {
      inline proc key(k) { return k; }
    }

    record KeysRanksComparator: keyComparator {
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
    private proc radixSortLSDCore(ref a:[?aD] ?t, nBits, negs, comparator) throws {
        try! rsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                       "type = %s nBits = %?".format(t:string,nBits));
        var temp = makeDistArray((...aD.shape), a.eltType);
        temp = a;

        // create a global count array to scan
        var globalCounts = makeDistArray((numLocales*numTasks*numBuckets), int);

        // loop over digits
        for rshift in {0..#nBits by bitsPerDigit} {
            const last = (rshift + bitsPerDigit) >= nBits;
            rsLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                           "rshift = ", rshift: string);
            // count digits
            coforall loc in Locales with (ref globalCounts) {
                on loc {
                    // allocate counts
                    var tasksBucketCounts: [Tasks] [0..#numBuckets] int;
                    coforall task in Tasks with (ref tasksBucketCounts) {
                        ref taskBucketCounts = tasksBucketCounts[task];
                        // get local domain's indices
                        var lD = temp.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
                        // count digits in this task's part of the array
                        for i in tD {
                            const key = comparator.key(temp.localAccess[i]);
                            var bucket = getDigit(key, rshift, last, negs); // calc bucket from key
                            taskBucketCounts[bucket] += 1;
                        }
                    }//coforall task
                    // write counts in to global counts in transposed order
                    coforall tid in Tasks with (ref tasksBucketCounts, ref globalCounts) {
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
                }//on loc
            }//coforall loc
            
            // scan globalCounts to get bucket ends on each locale/task
            var globalStarts = + scan globalCounts;
            globalStarts -= globalCounts;
            
            if vv {printAry("globalCounts =",globalCounts);try! stdout.flush();}
            if vv {printAry("globalStarts =",globalStarts);try! stdout.flush();}
            
            // calc new positions and permute
            coforall loc in Locales with (ref a) {
                on loc {
                    // allocate counts
                    var tasksBucketPos: [Tasks] [0..#numBuckets] int;
                    // read start pos in to globalStarts back from transposed order
                    coforall tid in Tasks with (ref tasksBucketPos) {
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
                    coforall task in Tasks with (ref tasksBucketPos, ref a) {
                        ref taskBucketPos = tasksBucketPos[task];
                        // get local domain's indices
                        var lD = temp.localSubdomain();
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
                }//on loc
            }//coforall loc

            // copy back to temp for next iteration
            // Only do this if there are more digits left
            if !last {
              temp <=> a;
            }
        } // for rshift
    }//proc radixSortLSDCore

    proc radixSortLSD(a:[?aD] ?t, checkSorted: bool = true): [aD] (t, int) throws {
        var kr: [aD] (t,int) = makeDistArray(aD, (t, int));
        kr = [(key,rank) in zip(a,aD)] (key,rank);
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
    proc radixSortLSD_ranks(a:[?aD] ?t, checkSorted: bool = true): [aD] int throws {
        if (checkSorted && isSorted(a)) {
            var ranks: [aD] int = [i in aD] i;
            return ranks;
        }

        var kr = makeDistArray(aD, (t, int));
        kr = [(key,rank) in zip(a,aD)] (key,rank);
        var (nBits, negs) = getBitWidth(a);
        radixSortLSDCore(kr, nBits, negs, new KeysRanksComparator());
        var ranks = makeDistArray(aD, int);
        ranks = [(_, rank) in kr] rank;
        return ranks;
    }

    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning sorted keys as a block distributed array */
    proc radixSortLSD_keys(a: [?aD] ?t, checkSorted: bool = true): [aD] t throws {
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
