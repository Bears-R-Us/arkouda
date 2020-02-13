module Merge {
  use SegmentedArray;
  use RadixSortLSD only numTasks, calcBlock;

  private config const DEBUG = false;
  
  /* Given a *sorted*, zero-up array, use binary search to find the index of the first element that is greater than or equal to a target.
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
    param BI = 1, BF = 2, SI = 3, SF = 4;
    var bounds: [LocaleSpace] [tD] 4*int;
    bounds[LocaleSpace.high][tD.high][BF] = big.size - 1;
    bounds[LocaleSpace.high][tD.high][SF] = small.size - 1;
    coforall loc in Locales {
      on loc {
        coforall task in tD {
          var bigPos = findStart(loc.id, task, big);
          bounds[loc.id][task][BI] = bigPos;
          var bigS = big[bigPos];
          var smallPos = binarySearch(small, bigS);
          bounds[loc.id][task][SI] = smallPos;
          // bounds[loc.id][task][OUTBASE] = big.offsets.a[bigPos] + small.offsets.a[smallPos];
          if task > 0 {
            bounds[loc.id][task-1][BF] = bigPos - 1;
            bounds[loc.id][task-1][SF] = smallPos - 1;
          } else if loc.id > 0 {
            bounds[loc.id-1][tD.high][BF] = bigPos - 1;
            bounds[loc.id-1][tD.high][SF] = smallPos - 1;
          }
        }
      }
    }
    // barrier
    if DEBUG {writeln("Starting second coforall");}
    coforall loc in Locales {
      on loc {
        if DEBUG {writeln("Inside loc"); stdout.flush();}
        ref biga = big.offsets.a;
        ref smalla = small.offsets.a;
        coforall task in tD {
          var bigPos = bounds[loc.id][task][BI];
          const bigEnd = bounds[loc.id][task][BF];
          var smallPos = bounds[loc.id][task][SI];
          const smallEnd = bounds[loc.id][task][SF];
          var outPos = bigPos + smallPos;
          const end = bounds[loc.id][task][BF] + bounds[loc.id][task][SF] + 1;
          var outOffset = biga[bigPos] + smalla[smallPos];
          var bigS = big[bigPos];
          var smallS = small[smallPos];
          if DEBUG && (numLocales == 1) { writeln("Task %t init: bigPos = %t, smallPos = %t, outPos = %t, end = %t, outOffset = %t, bigS = %s, smallS = %s".format(task, bigPos, smallPos, outPos, end, outOffset, bigS, smallS)); stdout.flush(); }
          // leapfrog the two arrays until all the output has been filled
          while outPos <= end {
            // take from the big array until it leapfrogs the small
            while (bigPos <= bigEnd) && ((smallPos > smallEnd) || (bigS <= smallS)) {
              if DEBUG {
                if (outPos > perm.domain.high) { writeln("OOB: outPos = %t not in %t".format(outPos, perm.domain)); stdout.flush();}
                if (bigPos > big.offsets.aD.high) { writeln("OOB: bigPos = %t not in %t".format(bigPos, big.offsets.aD)); stdout.flush();}
                if (outOffset + bigS.numBytes >= vals.size) { writeln("OOB: (outOffset = %t + bigS.numBytes = %t) not in %t".format(outOffset, bigS.numBytes, vals.domain)); stdout.flush();}
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
              if DEBUG {
                if (outPos > perm.domain.high) { writeln("OOB: outPos = %t not in %t".format(outPos, perm.domain)); stdout.flush();}
                if (smallPos > small.offsets.aD.high) { writeln("OOB: smallPos = %t not in %t".format(smallPos, small.offsets.aD)); stdout.flush();}
                if (outOffset + bigS.numBytes >= vals.size) { writeln("OOB: (outOffset = %t + smallS.numBytes = %t) not in %t".format(outOffset, smallS.numBytes, vals.domain)); stdout.flush();}
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