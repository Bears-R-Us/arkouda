/* unique finding and counting algorithms
 these are all based on dense histograms and sparse histograms(assoc domains/arrays)

 you could also use a sort if you got into a real bind with really
 large dense ranges of values and large arrays...

 *** need to factor in sparsity estimation somehow ***
 for example if (a.max-a.min > a.size) means that a's are sparse

 */
module Unique
{
    use ServerConfig;

    use Time;
    use Math only;

    //use PrivateDist;
    //use HashedDist;
    use BlockDist;

    use SymArrayDmap;

    use CommAggregation;
    use RadixSortLSD;
    use SegmentedString;
    use AryUtil;
    use Reflection;
    use Logging;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const uLogger = new Logger(logLevel, logChannel);

   /*
    sorting based unique finding procedure

    Returns a tuple: (UniqueValArray,UniqueValCountsArray)
    which contains the unique values of a, along with the number of times each unique value appears in a

    :arg a: Array of data to be processed
    :type a: [] int

    :returns: ([] int, [] int)
    */
    proc uniqueSort(a: [?aD] ?eltType, param needCounts = true) throws {
        if (aD.size == 0) {
            uLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"zero size");
            var u = makeDistArray(0, eltType);
            if (needCounts) {
                var c = makeDistArray(0, int);
                return (u,c);
            } else {
                return u;
            }
        } 

        var sorted = radixSortLSD_keys(a);
        return uniqueFromSorted(sorted, needCounts);
    }

    proc uniqueSortWithInverse(a: [?aD] ?eltType, param needIndices=false) throws {
        if (aD.size == 0) {
            uLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"zero size");
            var u = makeDistArray(aD.size, eltType);
            var c = makeDistArray(aD.size, int);
            var inv = makeDistArray(aD.size, int);
            var indices = makeDistArray(0, int);
            if needIndices
              then return (u, c, inv, indices);
              else return (u, c, inv);
        }
        var sorted = makeDistArray(aD, eltType);
        var perm = makeDistArray(aD, int);
        forall (s, p, sp) in zip(sorted, perm, radixSortLSD(a)) {
          (s, p) = sp;
        }
        var (u, c) = uniqueFromSorted(sorted);
        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
        overMemLimit(numBytes(int) * c.size);
        var segs = (+ scan c) - c;
        var bcast = makeDistArray(aD, int);
        forall s in segs with (var agg = newDstAggregator(int)) {
            agg.copy(bcast[s], 1);
        }
        bcast[0] = 0;
        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
        overMemLimit(numBytes(int) * bcast.size);
        bcast = (+ scan bcast);
        var inv = makeDistArray(aD.size, int);
        forall (p, b) in zip(perm, bcast) with (var agg = newDstAggregator(int)) {
            agg.copy(inv[p], b);
        }

        if needIndices {
          overMemLimit(numBytes(int) * u.size);
          var indices = makeDistArray(u.size, int);
          forall i in indices.domain with (var agg = newSrcAggregator(int)) {
            agg.copy(indices[i], perm[segs[i]]);
          }
          return (u, c, inv, indices);
        } else {
          return (u, c, inv);
        }
    }

    proc uniqueFromSorted(sorted: [?aD] ?eltType, param needCounts = true) throws {
        var truth = makeDistArray(aD, bool);
        truth[0] = true;
        [(t, s, i) in zip(truth, sorted, aD)] if i > aD.low { t = (sorted[i-1] != s); }
        var allUnique: int = + reduce truth;
        if (allUnique == aD.size) {
            uLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                    "early out already unique");
            var u = makeDistArray(aD.size, eltType);
            u = sorted; // array is already unique
            if (needCounts) {
                var c = makeDistArray(aD.size, int);
                c = 1;
                return (u,c);
            } else {
                return u;
            }
        }
        // +scan to compute segment position... 1-based because of inclusive-scan
        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
        overMemLimit(numBytes(int) * truth.size);
        var iv: [truth.domain] int = (+ scan truth);
        // compute how many segments
        var pop = iv[iv.size-1];
        uLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"pop = ",pop:string);

        var segs = makeDistArray(pop, int);
        var ukeys = makeDistArray(pop, eltType);
        
        // segment position... 1-based needs to be converted to 0-based because of inclusive-scan
        // where ever a segment break (true value) is... that index is a segment start index
        forall i in truth.domain with (var agg = newDstAggregator(int)) {
          if (truth[i] == true) {
            var idx = i; 
            agg.copy(segs[iv[i]-1], idx);
          }
        }
        // pull out the first key in each segment as a unique key
        // unique keys guaranteed to be sorted because keys are sorted
        forall (i, uk, seg) in zip(segs.domain, ukeys, segs) with (var agg = newSrcAggregator(eltType)) {
          agg.copy(uk, sorted[seg]);
        }
        // [i in segs.domain] ukeys[i] = sorted[segs[i]];

        if (needCounts) {
            var counts = makeDistArray(pop, int);
            // calc counts of each unique key using segs
            forall i in segs.domain {
                if i < segs.domain.high {
                    counts[i] = segs[i+1] - segs[i];
                }
                else
                {
                    counts[i] = sorted.domain.high+1 - segs[i];
                }
            }
            return (ukeys,counts);
        } else {
            return ukeys;
        }
    }

    proc uniqueGroup(str: SegString, returnInverse = false) throws {
        if (str.size == 0) {
            uLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"zero size");
            var uo = makeDistArray(0, int);
            var uv = makeDistArray(0, uint(8));
            var c = makeDistArray(0, int);
            var inv = makeDistArray(0, int);
            return (uo, uv, c, inv);
        }
        const ref aD = str.offsets.a.domain;
        var invD: aD.type;
        if returnInverse {
          invD = aD;
        } else {
          invD = {0..-1};
        }
        var inv = makeDistArray(invD, int);
        var truth = makeDistArray(aD, bool);
        var perm = makeDistArray(aD, int);
        if SegmentedStringUseHash {
          var hashes = str.siphash();
          var sorted = makeDistArray(aD, 2*uint);
          forall (s, p, sp) in zip(sorted, perm, radixSortLSD(hashes)) {
            (s, p) = sp;
          }
          truth[0] = true;
          [(t, s, i) in zip(truth, sorted, aD)] if i > aD.low { t = (sorted[i-1] != s); }
        } else {
          var soff = makeDistArray(aD, int);
          var sval = makeDistArray(str.values.a.domain, uint(8));
          perm = str.argsort();
          (soff, sval) = str[perm];
          truth[0] = true;
          // truth[{1..aD.high}] = sorted[0..aD.high-1] != sorted[1..aD.high];
          forall (t, o, idx) in zip(truth, soff, aD) {
            if (idx > aD.low) {
              const llen = o - soff[idx-1] - 1;
              const rlen = if (idx < aD.high) then (soff[idx+1] - 1 - o) else (sval.domain.high - o);
              if (llen != rlen) {
                // If lengths differ, this is a step
                t = true;
              } else {
                var allEqual = true;
                for pos in 0..#llen {
                  if (sval[soff[idx-1]+pos] != sval[o+pos]) {
                    allEqual = false;
                    break;
                  }
                }
                // If lengths equal but bytes differ, this is a step
                if !allEqual {
                  t = true;
                }
              }
            }
          }
        }
        var (uo, uv, c) = uniqueFromTruth(str, perm, truth);
        if returnInverse {
            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
            overMemLimit(numBytes(int) * c.size);
            var segs = (+ scan c) - c;
            var bcast = makeDistArray(invD, int);
            forall s in segs with (var agg = newDstAggregator(int)) {
                agg.copy(bcast[s], 1);
            }
            bcast[0] = 0;
            // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
            overMemLimit(numBytes(int) * bcast.size);
            bcast = (+ scan bcast);
            forall (p, b) in zip(perm, bcast) with (var agg = newDstAggregator(int)) {
                agg.copy(inv[p], b);
            }
        }
        return (uo, uv, c, inv);
    }

    proc uniqueFromTruth(str: SegString, perm: [?aD] int, truth: [aD] bool) throws {
        var allUnique: int = + reduce truth;
        if (allUnique == aD.size) {
            uLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"early out already unique");
            var uo = makeDistArray(aD.size, int);
            var uv = makeDistArray(str.nBytes, uint(8));
            var c = makeDistArray(aD.size, int);
            uo = str.offsets.a; // a is already unique
            uv = str.values.a;
            c = 1; // c counts are all 1
            return (uo, uv, c);
        }
        // +scan to compute segment position... 1-based because of inclusive-scan
        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
        overMemLimit(numBytes(int) * truth.size);
        var iv: [aD] int = (+ scan truth);
        // compute how many segments
        var pop = iv[iv.size-1];
        uLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"pop = %?".format(pop));

        var segs = makeDistArray(pop, int);
        var counts = makeDistArray(pop, int);
        var uinds = makeDistArray(pop, int);
        
        // segment position... 1-based needs to be converted to 0-based because of inclusive-scan
        // where ever a segment break (true value) is... that index is a segment start index
        forall i in aD with (var agg = newDstAggregator(int)) {
          if (truth[i] == true) {
            var idx = i;
            agg.copy(segs[iv[i]-1], idx);
          }
        }
        // pull out the first key in each segment as a unique key
        // unique keys guaranteed to be sorted because keys are sorted
        forall (u, s) in zip(uinds, segs) with (var agg = newSrcAggregator(int)) {
          agg.copy(u, perm[s]); // uinds[i] = perm[segs[i]];
        }
        // Gather the unique offsets and values (byte buffers)
        var (uo, uv) = str[uinds];
        // calc counts of each unique key using segs
        forall i in segs.domain {
            if i < segs.domain.high {
                counts[i] = segs[i+1] - segs[i];
            }
            else
            {
                counts[i] = aD.high+1 - segs[i];
            }
        }

        return (uo, uv, counts);
    }
}
