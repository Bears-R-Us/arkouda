module Merge {
  use IO;
  use SegmentedArray;
  use RadixSortLSD only numTasks, calcBlock;
  use Reflection;
  use ServerConfig;
  use Logging;
  
  const mLogger = new Logger();
  
  if v {
      mLogger.level = LogLevel.DEBUG;
  } else {
      mLogger.level = LogLevel.INFO;
  }
  
  /* Given a *sorted*, zero-up array, use binary search to find the index of the first element 
   * that is greater than or equal to a target.
   */
  proc binarySearch(a, x) throws {
    var l = 0;
    var r = a.size - 1;
    while l <= r {
      var mid = l + (r - l + 1)/2;
      if a[mid] < x {
        l = mid + 1;
      } else if a[mid] > x {
        r = mid - 1;
      } else { // this[mid] == s
        // Find the *first* match
        while (mid >= 0) && (a[mid] == x) {
          mid -= 1;
        }
        return mid + 1;
      } 
    }
    return l;
  }

  //const numTasks = RadixSortLSD.numTasks;
  inline proc findStart(loc, task, s: SegString) throws {
      ref va = s.values.a;
      const lD = va.localSubdomain();
      const tD = RadixSortLSD.calcBlock(task, lD.low, lD.high);
      const i = tD.low;
      ref oa = s.offsets.a;
      if && reduce (oa < i) {
          return va.size;
      } else {
          return binarySearch(oa, i);
      }    
  }

  proc mergeSorted(left: SegString, right: SegString) throws {
    const bigIsLeft = (left.size >= right.size);
    ref big = if bigIsLeft then left else right;
    ref small = if bigIsLeft then right else left;
    const size = big.size + small.size;
    const nBytes = big.nBytes + small.nBytes;
    var perm = makeDistArray(size, int);
    var segs = makeDistArray(size, int);
    var vals = makeDistArray(nBytes, uint(8));
    const tD = {0..#numTasks};
    // Element: (bigStart, bigStop, smallStart, smallStop)
    var bounds: [LocaleSpace] [tD] 4*int;
    bounds[LocaleSpace.high][tD.high] = (0, big.size-1, 0, small.size-1);
    coforall loc in Locales {
      on loc {
        coforall task in tD {
          ref (bigPos, _, smallPos, _) = bounds[loc.id][task];
          bigPos = findStart(loc.id, task, big);
          var bigS = big[bigPos];
          smallPos = binarySearch(small, bigS);
          ref (_, bigEnd, _, smallEnd) = if task > 0 then bounds[loc.id][task-1]
                                                     else bounds[loc.id-1][tD.high];
          bigEnd = bigPos - 1;
          smallEnd = smallPos - 1;
        }
      }
    }
    // barrier
    mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Starting second coforall");
    coforall loc in Locales {
      on loc {
          mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Inside loc"); 
          ref biga = big.offsets.a;
          ref smalla = small.offsets.a;
          coforall task in tD {
              var (bigPos, bigEnd, smallPos, smallEnd) = bounds[loc.id][task];
              var outPos = bigPos + smallPos;
              const end = bigEnd + smallEnd + 1;
              var outOffset = biga[bigPos] + smalla[smallPos];
              var bigS = big[bigPos];
              var smallS = small[smallPos];
              if (numLocales == 1) { 
                  mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                  "Task %t init: bigPos = %t, smallPos = %t, outPos = %t, end = %t, outOffset = %t, bigS = %s, smallS = %s".format(
                            task, bigPos, smallPos, outPos, end, outOffset, bigS, smallS)); 
              }
              // leapfrog the two arrays until all the output has been filled
              while outPos <= end {
                  // take from the big array until it leapfrogs the small
                  while (bigPos <= bigEnd) && ((smallPos > smallEnd) || (bigS <= smallS)) {
              
                      if (outPos > perm.domain.high) { 
                      mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                          "OOB: outPos = %t not in %t".format(outPos, perm.domain));
                  }
                  if (bigPos > big.offsets.aD.high) { 
                      mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                          "OOB: bigPos = %t not in %t".format(bigPos, big.offsets.aD));
                  }
                  if (outOffset + bigS.numBytes >= vals.size) { 
                      mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                           "OOB: (outOffset = %t + bigS.numBytes = %t) not in %t".format(
                                                               outOffset, bigS.numBytes, vals.domain));
                  }
                  if bigIsLeft {
                      perm[outPos] = bigPos;
                  } else {
                      perm[outPos] = bigPos + small.size;
                  }
                  segs[outPos] = outOffset;
                  const l = bigS.numBytes;
                  vals[{outOffset..#l}] = for b in bigS.chpl_bytes() do b;
                  outPos += 1;
                  outOffset += l + 1;
                  bigPos += 1;
                  if (bigPos <= bigEnd) {
                      bigS = big[bigPos];
                  }
             }
             // take from the small array until it catches up with the big
             while (smallPos <= smallEnd) && ((bigPos > bigEnd) || (smallS < bigS)) {          
                if (outPos > perm.domain.high) { 
                    mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                               "OOB: outPos = %t not in %t".format(outPos, perm.domain));
                }
                if (smallPos > small.offsets.aD.high) { 
                    mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                               "OOB: smallPos = %t not in %t".format(smallPos, small.offsets.aD));
                }
                if (outOffset + bigS.numBytes >= vals.size) {
                    mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                               "OOB: (outOffset = %t + smallS.numBytes = %t) not in %t".format(
                               outOffset, smallS.numBytes, vals.domain));
                }
            }
            if bigIsLeft {
                perm[outPos] = smallPos + big.size;
            } else {
                perm[outPos] = smallPos;
              }
              segs[outPos] = outOffset;
              const l = smallS.numBytes;
              vals[{outOffset..#l}] = for b in smallS.chpl_bytes() do b;
              outPos += 1;
              outOffset += l + 1;
              smallPos += 1;
              if (smallPos <= smallEnd) {
                smallS = small[smallPos];
              }
            }
          }
        }
      }
    }
    return (perm, segs, vals);
  }

  
}
