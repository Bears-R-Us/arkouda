/* Radix Sort Least Significant Digit */
module RadixSortLSD
{
    config const RSLSD_vv = true;
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

    
    inline proc getDigit(key: int, field: int, rshift: int): int {
        return ((key >> rshift) & maskDigit);
    }

    inline proc getDigit(key: ?t, field: int, rshift: int): t {
      return ((key[field] >> rshift) & maskDigit);
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

    inline proc copyElement(out dst: int, in src: int) {
      unorderedCopy(dst, src);
    }

    inline proc copyElement(out dst, in src) {
      for (d, s) in zip(dst, src) {
	unorderedCopy(d, s);
      }
    }

    inline proc getBitWidths(a: [] int): (int,) {
      var aMin = min reduce a;
      var aMax = max reduce a;
      var wPos = if aMax >= 0 then 64 - clz(aMax) else 0;
      var wNeg = if aMin < 0 then 65 - clz(-aMin) else 0;
      var ret: (int,);
      ret[1] = max(wPos, wNeg);
      return ret;
    }

    inline proc getBitWidths(a: [] ?t): t {
      var mins: t;
      var maxes: t;
      var bitWidths: t;
      forall tup in a with (min reduce mins, max reduce maxes) {
	for (field, i) in zip(tup, 1..) {
	  mins[i] = min(mins[i], field);
	  maxes[i] = max(maxes[i], field);
	}
      }
      for (lb, ub, w) in zip(mins, maxes, bitWidths) {
	var wPos = if ub >= 0 then 64 - clz(ub) else 0;
	var wNeg = if lb < 0 then 65 - clz(-lb) else 0;
	w = max(wPos, wNeg);
      }
      return bitWidths;
    }
	  
    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning a permutation vector also as a block distributed array */
    proc radixSortLSD(a: [?aD] ?t): [aD] int {

        // calc max value in bit position
        // *** need to fix this to take into account negative integers
      var bitWidths = getBitWidths(a);
        writeln("bitWidths = ", bitWidths);
        
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
	for field in bitWidths.size..1 by -1 {
	  const width = bitWidths[field];
	  if vv {writeln("field = ", field, ", width = ", width);}
	  for rshift in {0..#width by bitsPerDigit} {
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
			  var bucket = getDigit(kr0[i][KEY], field, rshift); // calc bucket from key
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
                        var lD = kr0.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
                        // read start pos in to globalStarts back from transposed order
                        for bucket in bD {
                            taskBucketPos[bucket] = globalStarts[calcGlobalIndex(bucket, loc.id, task)];
                        }
                        // calc new position and put (key,rank) pair there in kr1
                        for i in tD {
			  var bucket = getDigit(kr0[i][KEY], field, rshift); // calc bucket from key
                            var pos = taskBucketPos[bucket];
                            taskBucketPos[bucket] += 1;
                            // kr1[pos] = kr0[i];
                            //unorderedCopy(kr1[pos][KEY], kr0[i][KEY]);
			    copyElement(kr1[pos][KEY], kr0[i][KEY]);
                            unorderedCopy(kr1[pos][RANK], kr0[i][RANK]);
                        }
                    }//coforall task 
                }//on loc
            }//coforall loc

            // copy back to kr0 for next iteration
            kr0 = kr1;
            
	  }//for rshift
	  var negVal: int, firstNegative: int;
	  if (bitWidths.size > 1) {
	    (negVal, firstNegative) = minloc reduce zip([(key, rank) in kr1] key[field], aD);
	  } else {
	    (negVal, firstNegative) = minloc reduce zip([(key, rank) in kr1] key, aD);
	  }
	  [((key, rank), i) in zip(kr1[firstNegative..], aD.low..)] { copyElement(kr0[i][KEY], key); unorderedCopy(kr0[i][RANK], rank); }
	  [((key, rank), i) in zip(kr1[..firstNegative-1], aD.high-firstNegative+1..)] { copyElement(kr0[i][KEY], key); unorderedCopy(kr0[i][RANK], rank); }
	}// for (width, field)

        // find negative keys, they will appear in order at the high end of the array
        // if there are no negative keys then firstNegative will correspond to the lowest positive key in location 0
        /* var (negVal, firstNegative) = minloc reduce zip([(key,rank) in kr0] key, aD); */
        /* if vv then writeln((negVal,firstNegative)); */
        
        var ranks: [aD] int = [(key, rank) in kr0] rank;
        // copy the ranks corresponding to the negative keys to the beginning of the output array
        // [((key, rank), i) in zip(kr0[firstNegative..], aD.low..)] unorderedCopy(ranks[i], rank);
        // copy the ranks corresponding to the positive keys to the end of the output array
        // [((key, rank), i) in zip(kr0[..firstNegative-1], aD.high-firstNegative+1..)] unorderedCopy(ranks[i], rank);

        return ranks;
        
    }//proc radixSortLSD
    
}

