/* Radix Sort Least Significant Digit */
module RadixSortLSD
{
    config const RSLSD_vv = false;
    const vv = RSLSD_vv; // these need to be const for comms/performance reasons
    
    config const RSLSD_numTasks = here.maxTaskPar; // tasks per locale based on locale0
    const numTasks = RSLSD_numTasks; // tasks per locale
    const Tasks = {0..#numTasks}; // these need to be const for comms/performance reasons
    
    config param RSLSD_bitsPerDigit = 16;
    private param bitsPerDigit = RSLSD_bitsPerDigit; // these need to be const for comms/performance reasons
    private param numBuckets = 1 << bitsPerDigit; // these need to be const for comms/performance reasons
    private param maskDigit = numBuckets-1; // these need to be const for comms/performance reasons

    // Hack to determine tuple lower bound across 1.20-1.22
    const RSLSD_tupleLow = [1..1].domain.low;

    use BlockDist;
    use BitOps;
    use AryUtil;
    use CommAggregation;
    use IO;

    inline proc getBitWidth(a: [?aD] int): (int, bool) {
      var aMin = min reduce a;
      var aMax = max reduce a;
      var wPos = if aMax >= 0 then numBits(int) - clz(aMax) else 0;
      var wNeg = if aMin < 0 then numBits(int) - clz((-aMin)-1) + 1 else 0;
      const bitWidth = max(wPos, wNeg);
      const negs = aMin < 0;
      return (bitWidth, negs);
    }

    inline proc getBitWidth(a: [?aD] real): (int, bool) {
      const bitWidth = numBits(real);
      const negs = signbit(min reduce a);
      return (bitWidth, negs);
    }

    inline proc getBitWidth(a: [?aD] (uint, uint)): (int, bool) {
      const negs = false;
      var highMax = max reduce [(ai,_) in a] ai;
      var whigh = numBits(uint) - clz(highMax);
      if (whigh == 0) {
        var lowMax = max reduce [(_,ai) in a] ai;
        var wlow = numBits(uint) - clz(lowMax);
        const bitWidth = wlow: int;
        return (bitWidth, negs);
      } else {
        const bitWidth = (whigh + numBits(uint)): int;
        return (bitWidth, negs);
      }
    }

    inline proc getBitWidth(a: [?aD] ?t): (int, bool)
        where isHomogeneousTuple(t) && t == t.size*uint(bitsPerDigit) {
      for digit in 0..t.size-1 {
        const m = max reduce [ai in a] ai[RSLSD_tupleLow+digit];
        if m > 0 then return ((t.size-digit) * bitsPerDigit, false);
      }
      return (t.size * bitsPerDigit, false);
    }

    // Get the digit for the current rshift. In order to correctly sort
    // negatives, we have to invert the signbit if we're looking at the last
    // digit and the array contained negative values.
    inline proc getDigit(key: int, rshift: int, last: bool, negs: bool): int {
      const invertSignBit = last && negs;
      const xor = (invertSignBit:uint << (RSLSD_bitsPerDigit-1));
      const keyu = key:uint;
      return (((keyu >> rshift) & (maskDigit:uint)) ^ xor):int;
    }

    inline proc getDigit(key: uint, rshift: int, last: bool, negs: bool): int {
      return ((key >> rshift) & (maskDigit:uint)):int;
    }

    // Get the digit for the current rshift. In order to correctly sort
    // negatives, we have to invert the entire key if it's negative, and invert
    // just the signbit for positive values when looking at the last digit.
    inline proc getDigit(in key: real, rshift: int, last: bool, negs: bool): int {
      const invertSignBit = last && negs;
      var keyu: uint;
      c_memcpy(c_ptrTo(keyu), c_ptrTo(key), numBytes(key.type));
      var signbitSet = keyu >> (numBits(keyu.type)-1) == 1;
      var xor = 0:uint;
      if signbitSet {
        keyu = ~keyu;
      } else {
        xor = (invertSignBit:uint << (RSLSD_bitsPerDigit-1));
      }
      return (((keyu >> rshift) & (maskDigit:uint)) ^ xor):int;
    }

    inline proc getDigit(key: 2*uint, rshift: int, last: bool, negs: bool): int {
      const (key0,key1) = key;
      if (rshift >= numBits(uint)) {
        return getDigit(key0, rshift - numBits(uint), last, negs);
      } else {
        return getDigit(key1, rshift, last, negs);
      }
    }

    inline proc getDigit(key: _tuple, rshift: int, last: bool, negs: bool): int
        where isHomogeneousTuple(key) && key.type == key.size*uint(bitsPerDigit) {
      const keyHigh = key.size - (1-RSLSD_tupleLow);
      return key[keyHigh - rshift/bitsPerDigit]:int;
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
        
        var (nBits, negs) = getBitWidth(a);
        if vv {writeln("type = ", t:string, ", nBits = ", nBits);}
        
        // form (key,rank) vector
        var kr0: [aD] (t,int) = [(key,rank) in zip(a,aD)] (key,rank);
        var kr1: [aD] (t,int);
        
        // create a global count array to scan
        var gD = newBlockDom({0..#(numLocales * numTasks * numBuckets)});
        var globalCounts: [gD] int;
        var globalStarts: [gD] int;
        
        // loop over digits
        for rshift in {0..#nBits by bitsPerDigit} {
            const last = (rshift + bitsPerDigit) >= nBits;
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
                            const (key,_) = kr0[i];
                            var bucket = getDigit(key, rshift, last, negs); // calc bucket from key
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
                        {
                            var aggregator = newSrcAggregator(int);
                            for bucket in bD {
                                aggregator.copy(taskBucketPos[bucket], globalStarts[calcGlobalIndex(bucket, loc.id, task)]);
                            }
                            aggregator.flush();
                        }
                        // calc new position and put (key,rank) pair there in kr1
                        {
                            var aggregator = newDstAggregator((t,int));
                            for i in tD {
                                const (key,_) = kr0[i];
                                var bucket = getDigit(key, rshift, last, negs); // calc bucket from key
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
            if !last {
                kr0 = kr1;
            }
        } // for rshift

        var ranks: [aD] int = [(key, rank) in kr1] rank;
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
        var (nBits, negs) = getBitWidth(a);

        if vv {writeln("type = ", t:string, ", nBits = ", nBits);}
        
        var k0: [aD] t = a;
        var k1: [aD] t;
        
        // create a global count array to scan
        var gD = newBlockDom({0..#(numLocales * numTasks * numBuckets)});
        var globalCounts: [gD] int;
        var globalStarts: [gD] int;
        
        // loop over digits
        for rshift in {0..#nBits by bitsPerDigit} {
            const last = (rshift + bitsPerDigit) >= nBits;
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
                            var bucket = getDigit(k0[i], rshift, last, negs); // calc bucket from key
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
                        {
                            var aggregator = newSrcAggregator(int);
                            for bucket in bD {
                                aggregator.copy(taskBucketPos[bucket], globalStarts[calcGlobalIndex(bucket, loc.id, task)]);
                            }
                            aggregator.flush();
                        }
                        // calc new position and put (key,rank) pair there in kr1
                        {
                            var aggregator = newDstAggregator(t);
                            for i in tD {
                                var bucket = getDigit(k0[i], rshift, last, negs); // calc bucket from key
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
            if !last {
                k0 = k1;
            }
            
        }//for rshift
        return k1;
    }//proc radixSortLSD_keys

    proc radixSortLSD_memEst(size: int, itemsize: int) {
        return ((4 + 1) * size * itemsize)
                     + (2 * here.maxTaskPar * numLocales * 2**16 * 8);
    }
}

