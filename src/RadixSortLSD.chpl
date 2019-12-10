/* Radix Sort Least Significant Digit */
module RadixSortLSD
{
    config const RSLSD_vv = false;
    const vv = RSLSD_vv; // these need to be const for comms/performance reasons
    
    config const RSLSD_numTasks = here.maxTaskPar; // tasks per locale based on locale0
    const numTasks = RSLSD_numTasks; // tasks per locale
    const Tasks = {0..#numTasks}; // these need to be const for comms/performance reasons
    
    config const RSLSD_bitsPerDigit = 16;
    const bitsPerDigit = RSLSD_bitsPerDigit; // these need to be const for comms/performance reasons
    const numBuckets = 1 << bitsPerDigit; // these need to be const for comms/performance reasons
    const maskDigit = numBuckets-1; // these need to be const for comms/performance reasons

    // Select the mode for how small copies are initiated. For comm=none, there
    // is no reason to aggregate. For most other configurations aggregation is
    // always faster than unordered copy, so use it. Under comm=ugni, unordered
    // copy is faster than aggregation for collecting metadata about sorting
    // because the allocation and setup costs of aggregation costs are higher
    // than the comm benefit for the relatively small number of items copied.
    enum CopyMode {unordered, aggregated};
    private param noneOrUgni = CHPL_COMM == "ugni" || CHPL_COMM == "none";
    config const RSLSD_copyModeMeta = if noneOrUgni then CopyMode.unordered else CopyMode.aggregated;
    config const RSLSD_copyMode = if CHPL_COMM == "none" then CopyMode.unordered else CopyMode.aggregated;
    const copyModeMeta = RSLSD_copyModeMeta;
    const copyMode = RSLSD_copyMode;

    use BlockDist;
    use BitOps;
    use AryUtil;
    use UnorderedCopy;
    use CommAggregation;
    use IO;

    inline proc getBitWidth(a: [?aD] int): int {
      var aMin = min reduce a;
      var aMax = max reduce a;
      var wPos = if aMax >= 0 then numBits(int) - clz(aMax) else 0;
      var wNeg = if aMin < 0 then numBits(int) - clz(-aMin) + 1 else 0;
      wNeg = min(wNeg, numBits(int));
      return max(wPos, wNeg);
    }

    inline proc getBitWidth(a: [?aD] real): int{
      return numBits(real);
    }

    inline proc getBitWidth(a: [?aD] (uint, uint)): int {
      var highMax = max reduce [ai in a] ai[1];
      var whigh = numBits(uint) - clz(highMax);
      if (whigh == 0) {
        var lowMax = max reduce [ai in a] ai[2];
        var wlow = numBits(uint) - clz(lowMax);
        return wlow: int;
      } else {
        return (whigh + numBits(uint)): int;
      }
    }
    
    inline proc getDigit(key: int, rshift: int): int {
        return ((key >> rshift) & maskDigit);
    }

    inline proc realToUint(in r: real): uint {
        var u: uint;
        c_memcpy(c_ptrTo(u), c_ptrTo(r), numBytes(r.type));
        return u;
    }

    inline proc getDigit(key: real, rshift: int): int {
        var keyu = realToUint(key);
        return ((keyu >> rshift) & maskDigit):int;
    }

    inline proc isNeg(key) {
      if isReal(key) {
        return signbit(key);
      } else {
        return key < 0;
      }
    }

    inline proc getDigit(key: 2*uint, rshift: int): int {
      if (rshift >= numBits(uint)) {
        return getDigit(key[1], rshift - numBits(uint));
      } else {
        return getDigit(key[2], rshift);
      }
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
       radix sort a block distributed array
       returning a permutation vector as a block distributed array */
    proc radixSortLSD_ranks(a:[?aD] ?t, checkSorted: bool = true): [aD] int {

        // check to see if array is already sorted
        if (checkSorted) {
            if (isSorted(a)) {
                var ranks: [aD] int = [i in aD] i;
                return ranks;
            }
        }
        
        var nBits = getBitWidth(a);
        if vv {writeln("type = ", t:string, ", nBits = ", nBits);}
        
        // form (key,rank) vector
        param KEY = 1; // index of key in pair
        param RANK = 2; // index of rank in pair
        var kr0: [aD] (t,int) = [(key,rank) in zip(a,aD)] (key,rank);
        var kr1: [aD] (t,int);
        
        // create a global count array to scan
        var gD = newBlockDom({0..#(numLocales * numTasks * numBuckets)});
        var globalCounts: [gD] int;
        var globalStarts: [gD] int;
        
        // loop over digits
        for rshift in {0..#nBits by bitsPerDigit} {
            if vv {writeln("rshift = ",rshift);}
            // count digits
            coforall loc in Locales {
                on loc {
                    coforall task in Tasks {
                        // bucket domain
                        var bD = {0..#numBuckets};
                        // allocate counts
                        var taskBucketCounts: [bD] int;
                        // get local domain's indices
                        var lD = kr0.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
                        if vv {writeln((loc.id,task,tD));}
                        // count digits in this task's part of the array
                        for i in tD {
                            var bucket = getDigit(kr0[i][KEY], rshift); // calc bucket from key
                            taskBucketCounts[bucket] += 1;
                        }
                        // write counts in to global counts in transposed order
			if copyModeMeta == CopyMode.unordered {
			    for bucket in bD {
				//globalCounts[calcGlobalIndex(bucket, loc.id, task)] = taskBucketCounts[bucket];
				// will/does this make a difference???
				unorderedCopy(globalCounts[calcGlobalIndex(bucket, loc.id, task)], taskBucketCounts[bucket]);
			    }
			    unorderedCopyTaskFence();
                        } else if copyModeMeta == CopyMode.aggregated {
			    var aggregator = new DstAggregator(int);
			    for bucket in bD {
				aggregator.copy(globalCounts[calcGlobalIndex(bucket, loc.id, task)], taskBucketCounts[bucket]);
			    }
			    aggregator.flush();
                        }
                    }//coforall task
                }//on loc
            }//coforall loc
            
            // scan globalCounts to get bucket ends on each locale/task
            globalStarts = + scan globalCounts;
            globalStarts = globalStarts - globalCounts;
            
            if vv {printAry("globalCounts =",globalCounts);try! stdout.flush();}
            if vv {printAry("globalStarts =",globalStarts);try! stdout.flush();}
            
            // calc new positions and permute
            coforall loc in Locales {
                on loc {
                    coforall task in Tasks {
                        // bucket domain
                        var bD = {0..#numBuckets};
                        // allocate counts
                        var taskBucketPos: [bD] int;
                        // get local domain's indices
                        var lD = kr0.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
                        // read start pos in to globalStarts back from transposed order
			if copyModeMeta == CopyMode.unordered {
			    for bucket in bD {
				//taskBucketPos[bucket] = globalStarts[calcGlobalIndex(bucket, loc.id, task)];
				// will/does this make a difference???
				unorderedCopy(taskBucketPos[bucket], globalStarts[calcGlobalIndex(bucket, loc.id, task)]);
			    }
			    unorderedCopyTaskFence();
                        } else if copyModeMeta == CopyMode.aggregated {
                            var aggregator = new SrcAggregator(int);
                            for bucket in bD {
                                aggregator.copy(taskBucketPos[bucket], globalStarts[calcGlobalIndex(bucket, loc.id, task)]);
                            }
                            aggregator.flush();
                        }
                        // calc new position and put (key,rank) pair there in kr1
			if copyMode == CopyMode.unordered {
                            for i in tD {
				var bucket = getDigit(kr0[i][KEY], rshift); // calc bucket from key
                                var pos = taskBucketPos[bucket];
                                taskBucketPos[bucket] += 1;
                                // kr1[pos] = kr0[i];
                                if isTuple(t) {
                                  for param elem in 1..t.size {
                                    unorderedCopy(kr1[pos][KEY][elem], kr0[i][KEY][elem]);
                                  }
                                } else {
                                  unorderedCopy(kr1[pos][KEY],  kr0[i][KEY]);
                                }
                                unorderedCopy(kr1[pos][RANK], kr0[i][RANK]);
                            }
                            unorderedCopyTaskFence();
                        } else if copyMode == CopyMode.aggregated {
			    var aggregator = new DstAggregator((t,int));
			    for i in tD {
				var bucket = getDigit(kr0[i][KEY], rshift); // calc bucket from key
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
	    if (rshift + bitsPerDigit) < nBits {
                kr0 = kr1;
	    }
        } // for rshift
        
	// find negative keys, they will appear together at the high end of the array
	// if there are no negative keys then firstNegative will be aD.low
        var hasNegatives: bool , firstNegative: int = aD.high + 1;
        // maxloc on bools returns the first index where condition is true
        if !isTuple(t) {
          // For now, the assumption is that tuples contain hashes and are unsigned
          // We will need additional logic if we want to support arbitrary tuples
          (hasNegatives, firstNegative) = maxloc reduce zip([(key,rank) in kr1] (key < 0), aD);
        }
        // Swap the ranks of the positive and negative keys, so that negatives come first
        // If real type, then negative keys will appear in descending order and
        // must be reversed
        const negStride = if (isRealType(t) && hasNegatives) then -1 else 1;
        const numNegatives = aD.high - firstNegative + 1;
        if vv {writeln("hasNegatives? ", hasNegatives, ", negStride = ", negStride,
                       ", firstNegative = ", firstNegative, ", numNegatives = ", numNegatives);}
        
        var ranks: [aD] int;
        // Copy negatives to the beginning
        [((key, rank), i) in zip(kr1[firstNegative..], aD.low..aD.low+numNegatives-1 by negStride)] unorderedCopy(ranks[i], rank);
        // Copy positives to the end
        [((key, rank), i) in zip(kr1[..firstNegative-1], aD.low+numNegatives..)] unorderedCopy(ranks[i], rank);
        // No need to copy keys, because we are only returning ranks
        
        
        return ranks;
        
    }//proc radixSortLSD_ranks
    

    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning sorted keys as a block distributed array */
    proc radixSortLSD_keys(a: [?aD] ?t, checkSorted: bool = true): [aD] t {

        // check to see if array is already sorted
        if (checkSorted) {
            if (isSorted(a)) {
                var sorted: [aD] t = a;
                return sorted;
            }
        }
        
        // calc max value in bit position
        var nBits = getBitWidth(a);
        if vv {writeln("type = ", t:string, ", nBits = ", nBits);}
        
        var k0: [aD] t = a;
        var k1: [aD] t;
        
        // create a global count array to scan
        var gD = newBlockDom({0..#(numLocales * numTasks * numBuckets)});
        var globalCounts: [gD] int;
        var globalStarts: [gD] int;
        
        // loop over digits
        for rshift in {0..#nBits by bitsPerDigit} {
            if vv {writeln("rshift = ",rshift);}
            // count digits
            coforall loc in Locales {
                on loc {
                    coforall task in Tasks {
                        // bucket domain
                        var bD = {0..#numBuckets};
                        // allocate counts
                        var taskBucketCounts: [bD] int;
                        // get local domain's indices
                        var lD = k0.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
                        if vv {writeln((loc.id,task,tD));}
                        // count digits in this task's part of the array
                        for i in tD {
                            var bucket = getDigit(k0[i], rshift); // calc bucket from key
                            taskBucketCounts[bucket] += 1;
                        }
                        // write counts in to global counts in transposed order
			if copyModeMeta == CopyMode.unordered {
			    for bucket in bD {
				//globalCounts[calcGlobalIndex(bucket, loc.id, task)] = taskBucketCounts[bucket];
				// will/does this make a difference???
				unorderedCopy(globalCounts[calcGlobalIndex(bucket, loc.id, task)], taskBucketCounts[bucket]);
			    }
			    unorderedCopyTaskFence();
                        } else if copyModeMeta == CopyMode.aggregated {
			    var aggregator = new DstAggregator(int);
			    for bucket in bD {
				aggregator.copy(globalCounts[calcGlobalIndex(bucket, loc.id, task)], taskBucketCounts[bucket]);
			    }
			    aggregator.flush();
                        }
                    }//coforall task
                }//on loc
            }//coforall loc
            
            // scan globalCounts to get bucket ends on each locale/task
            globalStarts = + scan globalCounts;
            globalStarts = globalStarts - globalCounts;
            
            if vv {printAry("globalCounts =",globalCounts);try! stdout.flush();}
            if vv {printAry("globalStarts =",globalStarts);try! stdout.flush();}
            
            // calc new positions and permute
            coforall loc in Locales {
                on loc {
                    coforall task in Tasks {
                        // bucket domain
                        var bD = {0..#numBuckets};
                        // allocate counts
                        var taskBucketPos: [bD] int;
                        // get local domain's indices
                        var lD = k0.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
                        // read start pos in to globalStarts back from transposed order
			if copyModeMeta == CopyMode.unordered {
			    for bucket in bD {
				//taskBucketPos[bucket] = globalStarts[calcGlobalIndex(bucket, loc.id, task)];
				// will/does this make a difference???
				unorderedCopy(taskBucketPos[bucket], globalStarts[calcGlobalIndex(bucket, loc.id, task)]);
			    }
			    unorderedCopyTaskFence();
                        } else if copyModeMeta == CopyMode.aggregated {
                            var aggregator = new SrcAggregator(int);
                            for bucket in bD {
                                aggregator.copy(taskBucketPos[bucket], globalStarts[calcGlobalIndex(bucket, loc.id, task)]);
                            }
                            aggregator.flush();
                        }
                        // calc new position and put (key,rank) pair there in kr1
			if copyMode == CopyMode.unordered {
                            for i in tD {
                                var bucket = getDigit(k0[i], rshift); // calc bucket from key
                                var pos = taskBucketPos[bucket];
                                taskBucketPos[bucket] += 1;
                                // k1[pos] = k0[i];
				unorderedCopy(k1[pos], k0[i]);
                            }
                            unorderedCopyTaskFence();
                        } else if copyMode == CopyMode.aggregated {
			    var aggregator = new DstAggregator(t);
                            for i in tD {
                                var bucket = getDigit(k0[i], rshift); // calc bucket from key
                                var pos = taskBucketPos[bucket];
                                taskBucketPos[bucket] += 1;
				aggregator.copy(k1[pos], k0[i]);
                            }
                            aggregator.flush();
                        }
                    }//coforall task 
                }//on loc
            }//coforall loc
            
            // copy back to k0 for next iteration
            // Only do this if there are more digits left
	    // If this is the last digit, the negative-swapping code will copy the ranks
	    if (rshift + bitsPerDigit) < nBits {
                k0 = k1;
	    }
            
        }//for rshift
        
	// find negative keys, they will appear together at the high end of the array
        // if there are no negative keys then firstNegative will be aD.low
        var hasNegatives: bool , firstNegative: int;
        // maxloc on bools returns the first index where condition is true
        (hasNegatives, firstNegative) = maxloc reduce zip([key in k1] (isNeg(key)), aD);
        // Swap the ranks of the positive and negative keys, so that negatives come first
        // If real type, then negative keys will appear in descending order and
        // must be reversed
        const negStride = if (isRealType(t) && hasNegatives) then -1 else 1;
        const numNegatives = aD.high - firstNegative + 1;
        if vv {writeln("hasNegatives? ", hasNegatives, ", negStride = ", negStride,
                       ", firstNegative = ", firstNegative, ", numNegatives = ", numNegatives);}
        // Copy negatives to the beginning
        [(key, i) in zip(k1[firstNegative..], aD.low..aD.low+numNegatives-1 by negStride)] unorderedCopy(k0[i], key);
        // Copy positives to the end
        [(key, i) in zip(k1[..firstNegative-1], aD.low+numNegatives..)] unorderedCopy(k0[i], key);
        
        return k0;
        
    }//proc radixSortLSD_keys
    
}

