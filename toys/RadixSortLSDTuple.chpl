/* Radix Sort Least Significant Digit */
module RadixSortLSDTuple
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

    inline proc getDigit(key: int, rshift: int): int {
        return ((key >> rshift) & maskDigit);
    }
    
    proc getDigit(key: ?t, field: int, rshift: int): int where isIntegral(t) {
        return ((key >> rshift) & maskDigit);
    }

    extern {
      unsigned long long shiftDouble(double key, long long rshift) {
        // Reinterpret the bits of key as an unsigned 64-bit int (u long long)
        // Unsigned because we want to left-extend with zeros
        unsigned long long intkey = * (unsigned long long *)&key;
        return (intkey >> rshift);
      }
    }
    
    inline proc getDigit(key: real, rshift: int): int {
      var shiftedKey: uint = shiftDouble(key: c_double, rshift: c_longlong): uint;
      return (shiftedKey & maskDigit):int;
    }
    
    proc getDigit(key: ?t, field: int, rshift: int): int where isRealType(t) {
      var shiftedKey: uint = shiftDouble(key: c_double, rshift: c_longlong): uint;
      return (shiftedKey & maskDigit):int;
    }

    proc getDigit(key: ?t , field: int, rshift: int): int where isTuple(t) {
      //return ((key[field] >> rshift) & maskDigit);
      for param i in 1..t.size {
        if (i == field) {
          return getDigit(key[i], rshift);
        }
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

    inline proc copyElement(out dst: ?t, in src: t) where isNumeric(t) {
      unorderedCopy(dst, src);
    }

    inline proc copyElement(out dst: ?t, in src: t) where isTuple(t) {
      for param i in 1..t.size {
        unorderedCopy(dst[i], src[i]);
      }
    }

    inline proc getBitWidths(a: [] ?t): (int,) where isIntegral(t) {
      var aMin = min reduce a;
      var aMax = max reduce a;
      var wPos = if aMax >= 0 then numBits(a.eltType) - clz(aMax) else 0;
      var wNeg = if aMin < 0 then numBits(a.eltType) - clz(-aMin) + 1 else 0;
      var ret: (int,);
      ret[1] = max(wPos, wNeg);
      return ret;
    }

    inline proc getBitWidths(a: [] ?t): (int,) where isReal(t) {
      var ret: (int,);
      ret[1] = numBits(t);
      return ret;
    }

    inline proc getBitWidths(a: [] ?t): (t.size*int) where isTuple(t) {
      type rt = t.size * int;
      const firstElem = a[a.domain.low];
      var mins: t;
      var maxes: t;
      for param i in 1..t.size {
        mins[i] = max(firstElem[i].type);
        maxes[i] = min(firstElem[i].type);
      }
      var bitWidths: rt;
      forall tup in a with (ref mins, ref maxes) {
        for param i in 1..t.size {
          mins[i] = min(mins[i], tup[i]);
          maxes[i] = max(maxes[i], tup[i]);
        }
      }
      for param i in 1..t.size {
        // (lb, ub, w) in zip(mins, maxes, bitWidths) {
        if isRealType(firstElem[i].type) {
          bitWidths[i] = numBits(firstElem[i].type);
        } else if isIntegral(firstElem[i].type) {
          var wPos = if maxes[i] >= 0 then numBits(firstElem[i].type) - clz(maxes[i]) else 0;
          var wNeg = if mins[i] < 0 then numBits(firstElem[i].type) - clz(-mins[i]) + 1 else 0;
          bitWidths[i] = max(wPos, wNeg);
        }
      }
      return bitWidths;
    }
          
    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning a permutation vector also as a block distributed array */
    proc radixSortLSD(a: [?aD] ?t): [aD] int {
      var ranks: [aD] int;
      if (a.size == 0) { return ranks; }
      if (a.size == 1) { ranks[aD.low] = aD.low; return ranks; }
      // Make a tuple for checking field types later
      const firstElem = if isTuple(t) then a[aD.low] else (a[aD.low],);
      if vv { writeln("field types = ", firstElem.type:string); }
      // Make a tuple of num bits to sort for each field
      const bitWidths = getBitWidths(a);
      if vv {writeln("bitWidths = ", bitWidths);}
        
        // form (key,rank) vector
        param KEY = 1; // index of key in pair
        param RANK = 2; // index of rank in pair
        var kr0: [aD] (t,int) = [(key,rank) in zip(a,aD)] (key,rank);
        var kr1: [aD] (t,int);

        // create a global count array to scan
        var gD = newBlockDom({0..#(numLocales * numTasks * numBuckets)});
        var globalCounts: [gD] int;
        var globalStarts: [gD] int;

        // loop over tuple fields
        for param field in 1..bitWidths.size by -1 {
          const width = bitWidths[field];
          if vv {writeln("field = ", field, ", type = ", firstElem[field].type:string, ", width = ", width);}
          // loop over digits
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
                        // if vv {writeln((loc.id,task,tD));}
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
            // Only do this if there are more digits left in this tuple element
            // If this is the end of the tuple element, the negative-swapping code will perform the copy
            if (rshift + bitsPerDigit) < width {
              kr0 = kr1;
            }
          }//for rshift
          
          // At the end of every tuple element, check for negative keys
          // Negative keys will appear in order at the high end of the array
          // if there are no negative keys then firstNegative will correspond to the lowest positive key in location 0
          var hasNegatives: bool , firstNegative: int;
          if (bitWidths.size > 1) { // elements are tuples
            (hasNegatives, firstNegative) = maxloc reduce zip([(key, rank) in kr1] (key[field] < 0), aD);
          } else { // elements are ints or reals
            (hasNegatives, firstNegative) = maxloc reduce zip([(key, rank) in kr1] (key < 0), aD);
          }
          // Swap the positive and negative keys, so that negatives come first
          // If real type, then negative keys will appear in descending order and must be reversed
          const negStride = if (isRealType(firstElem[field].type) && hasNegatives) then -1 else 1;
          const numNegatives = aD.high - firstNegative + 1;
          if vv {writeln("hasNegatives? ", hasNegatives, ", negStride = ", negStride, ", firstNegative = ", firstNegative, ", numNegatives = ", numNegatives);}
          // Copy negatives to the beginning
          [((key, rank), i) in zip(kr1[firstNegative..], aD.low..aD.low+numNegatives-1 by negStride)] { copyElement(kr0[i][KEY], key); unorderedCopy(kr0[i][RANK], rank); }
          // Copy positives to the end
          [((key, rank), i) in zip(kr1[..firstNegative-1], aD.low+numNegatives..)] { copyElement(kr0[i][KEY], key); unorderedCopy(kr0[i][RANK], rank); }
        }// for (width, field)

        // Output just the ranks for argsort
        ranks = [(key, rank) in kr0] rank;
        return ranks;
        
    }//proc radixSortLSD
    
}

