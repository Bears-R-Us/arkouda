/* Radix Sort Least Significant Digit */
module RadixSortLSD
{
    config const RSLSD_vv = false;
    var vv = RSLSD_vv;
    
    config const RSLSD_numTasks = here.maxTaskPar; // tasks per locale based on locale0
    var numTasks = RSLSD_numTasks; // tasks per locale
    var Tasks = {0..#numTasks};
    
    config const RSLSD_bitsPerDigit = 16;
    var bitsPerDigit = RSLSD_bitsPerDigit;
    var numBuckets = 1 << bitsPerDigit;
    var maskDigit = numBuckets-1;

    use BlockDist;
    use BitOps;
    use AryUtil;
    use UnorderedCopy;

    inline proc getBitWidth(a: [] ?t): int where isIntegral(t) {
      var aMin = min reduce a;
      var aMax = max reduce a;
      var wPos = if aMax >= 0 then numBits(t) - clz(aMax) else 0;
      var wNeg = if aMin < 0 then numBits(t) - clz(-aMin) + 1 else 0;
      return max(wPos, wNeg);
    }

    inline proc getBitWidth(a: [] ?t): int where isReal(t) {
      return numBits(t);
    }
    
    inline proc getDigit(key: int, rshift: int): int {
        return ((key >> rshift) & maskDigit);
    }

    extern {
      static unsigned long long shiftDouble(double key, long long rshift) {
	// Reinterpret the bits of key as an unsigned 64-bit int (u long long)
	// Unsigned because we want to left-extend with zeros
	unsigned long long intkey = * (unsigned long long *) &key;
	return (intkey >> rshift);
      }
    }
    
    inline proc getDigit(key: real, rshift: int): int {
      var shiftedKey: uint = shiftDouble(key: c_double, rshift: c_longlong): uint;
      return (shiftedKey & maskDigit):int;
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
    proc radixSortLSD_ranks(a:[?aD] ?t): [aD] int {
      var nBits = getBitWidth(a);
      if vv {writeln("type = ", t:string, ", nBits = ", nBits);}
      
      var r0: [aD] int = [rank in aD] rank;
      var r1: [aD] int;
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
                        for bucket in bD {
                            globalCounts[calcGlobalIndex(bucket, loc.id, task)] = taskBucketCounts[bucket];
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
                        for bucket in bD {
                            taskBucketPos[bucket] = globalStarts[calcGlobalIndex(bucket, loc.id, task)];
                        }
                        // calc new position and put (key,rank) pair there in kr1
                        for i in tD {
                            var bucket = getDigit(k0[i], rshift); // calc bucket from key
                            var pos = taskBucketPos[bucket];
                            taskBucketPos[bucket] += 1;
                            // kr1[pos] = kr0[i];
                            unorderedCopy(k1[pos], k0[i]);
                            unorderedCopy(r1[pos], r0[i]);
                        }
                    }//coforall task 
                }//on loc
            }//coforall loc

            // copy back to k0 and r0 for next iteration
	    // Only do this if there are more digits left
	    // If this is the last digit, the negative-swapping code will copy the ranks
	    if (rshift + bitsPerDigit) < nBits {
	      k0 = k1;
	      r0 = r1;
	    }
	  } // for rshift

	  // find negative keys, they will appear together at the high end of the array
	  // if there are no negative keys then firstNegative will be aD.low
	  var hasNegatives: bool , firstNegative: int;
	  // maxloc on bools returns the first index where condition is true
	  (hasNegatives, firstNegative) = maxloc reduce zip([key in k1] (key < 0), aD);
	  // Swap the ranks of the positive and negative keys, so that negatives come first
	  // If real type, then negative keys will appear in descending order and
	  // must be reversed
	  const negStride = if (isRealType(t) && hasNegatives) then -1 else 1;
	  const numNegatives = aD.high - firstNegative + 1;
	  if vv {writeln("hasNegatives? ", hasNegatives, ", negStride = ", negStride,
			 ", firstNegative = ", firstNegative, ", numNegatives = ", numNegatives);}
	  // Copy negatives to the beginning
	  [(rank, i) in zip(r1[firstNegative..], aD.low..aD.low+numNegatives-1 by negStride)] unorderedCopy(r0[i], rank);
	  // Copy positives to the end
	  [(rank, i) in zip(r1[..firstNegative-1], aD.low+numNegatives..)] unorderedCopy(r0[i], rank);
	  // No need to copy keys, because we are only returning ranks
	
        return r0;
        
    }//proc radixSortLSD_ranks
    

    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning sorted keys as a block distributed array */
    proc radixSortLSD_keys(a: [?aD] ?t): [aD] t {
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
                        for bucket in bD {
                            globalCounts[calcGlobalIndex(bucket, loc.id, task)] = taskBucketCounts[bucket];
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
                        for bucket in bD {
                            taskBucketPos[bucket] = globalStarts[calcGlobalIndex(bucket, loc.id, task)];
                        }
                        // calc new position and put (key,rank) pair there in kr1
                        for i in tD {
                            var bucket = getDigit(k0[i], rshift); // calc bucket from key
                            var pos = taskBucketPos[bucket];
                            taskBucketPos[bucket] += 1;
                            // k1[pos] = k0[i];
                            unorderedCopy(k1[pos], k0[i]);
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
	  (hasNegatives, firstNegative) = maxloc reduce zip([key in k1] (key < 0), aD);
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

