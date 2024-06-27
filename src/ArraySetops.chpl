/* Array set operations
 includes intersection, union, xor, and diff

 currently, only performs operations with integer arrays 
 */

module ArraySetops
{
    use ServerConfig;
    use Logging;

    use SymArrayDmap;
    use RadixSortLSD;
    use Unique;
    use Indexing;
    use AryUtil;
    use In1d;
    use CommAggregation;
    use PrivateDist;
    use Search;
    use Sort.TwoArrayRadixSort;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const asLogger = new Logger(logLevel, logChannel);

    // returns intersection of 2 arrays
    proc intersect1d(a: [] ?t, b: [] t, assume_unique: bool) throws {
      //if not unique, unique sort arrays then perform operation
      if (!assume_unique) {
        var a1  = uniqueSort(a, false);
        var b1  = uniqueSort(b, false);
        return intersect1dHelper(a1, b1);
      }
      return intersect1dHelper(a,b);
    }

    proc intersect1dHelper(a: [] ?t, b: [] t) throws {
      var aux = radixSortLSD_keys(concatArrays(a,b));

      // All elements except the last
      const ref head = aux[..aux.domain.high-1];

      // All elements except the first
      const ref tail = aux[aux.domain.low+1..];
      const mask = head == tail;

      return boolIndexer(head, mask);
    }
    
    // returns the exclusive-or of 2 arrays
    proc setxor1d(a: [] ?t, b: [] t, assume_unique: bool) throws {
      //if not unique, unique sort arrays then perform operation
      if (!assume_unique) {
        var a1  = uniqueSort(a, false);
        var b1  = uniqueSort(b, false);
        return  setxor1dHelper(a1, b1);
      }
      return setxor1dHelper(a,b);
    }

    // Gets xor of 2 arrays
    // first concatenates the 2 arrays, then
    // sorts and removes all values that occur
    // more than once
    proc setxor1dHelper(a: [] ?t, b: [] t) throws {
      const aux = radixSortLSD_keys(concatArrays(a,b));
      const ref D = aux.domain;

      // Concatenate a `true` onto each end of the array
      var flag = makeDistArray(aux.size+1, bool);
      const ref fD = flag.domain;
      
      flag[fD.low] = true;
      flag[fD.low+1..fD.high-1] = aux[..D.high-1] != aux[D.low+1..];
      flag[fD.high] = true;

      var mask;
      {
        mask = sliceTail(flag) & sliceHead(flag);
      }

      var ret = boolIndexer(aux, mask);

      return ret;
    }

    // returns the set difference of 2 arrays
    proc setdiff1d(ref a: [] ?t, ref b: [] t, assume_unique: bool) throws {
      //if not unique, unique sort arrays then perform operation
      if (!assume_unique) {
        var a1  = uniqueSort(a, false);
        var b1  = uniqueSort(b, false);
        return setdiff1dHelper(a1, b1);
      }
      return setdiff1dHelper(a,b);
    }
    
    // Gets diff of 2 arrays
    // first checks membership of values in
    // fist array in second array and stores
    // as a boolean array and inverts these
    // values and returns the array indexed
    // with this inverted array
    proc setdiff1dHelper(ref a: [] ?t, ref b: [] t) throws {
        var truth = in1d(a, b, invert=true);
        var ret = boolIndexer(a, truth);
        return ret;
    }
    
    // Gets union of 2 arrays
    // first concatenates the 2 arrays, then
    // sorts resulting array and ensures that
    // values are unique
    proc union1d(a: [] ?t, b: [] t) throws {
      var aux;
      // Artificial scope to clean up temporary arrays
      {
        aux = concatArrays(uniqueSort(a,false), uniqueSort(b,false));
      }
      return uniqueSort(aux, false);
    }

    proc mergeHelper(ref sortedIdx: [?sD] ?t, ref permutedVals: [] ?t2, const ref idx1: [?D] t, const ref idx2: [] t, const ref val1: [] t2, const ref val2: [] t2, percentTransferLimit:int = 100) throws {
      const allocSize = sD.size;
      // create refs to arrays so it's easier to swap them if we come up with a good heuristic
      //  for which causes fewer data shuffles between locales
      const ref a = idx1;
      const ref aVal = val1;
      const ref b = idx2;
      const ref bVal = val2;

      // private space is a distributed domain of size numLocales
      var segs: [PrivateSpace] int;
      // indicates if we need to pull any values from the next locale.
      // the first bool is for a and the second is for b
      var pullLocalFlag: [0..<numLocales] (bool, bool);
      // number of elements to pull local (we only need one int because only one of a and b will ever need to
      //  fetch from the next locale). proofish:
      //  a and b are sorted, so a_max_loc_i <= a_min_loc_(i+1) and b_max_loc_i <= b_min_loc_(i+1)
      //  assume fetch condition a_min_loc_(i+1) < b_max_loc_i is met, combining these inequalities gives
      //  a_max_loc_i <= a_min_loc_(i+1) < b_max_loc_i <= b_min_loc_(i+1) => the other fetch condition cannot be met
      var pullLocalCount: [0..<numLocales] int;

      // if the merge based workflow seems too comm heavy, fall back to radix sort
      var toGiveUp = false;

      var aMin = -1, aMax = -1, bMin = -1, bMax = -1;
      // we iterate over locales serially because the logic relies on the mins and maxs referring to the previous locale
      // use an atomic counter to ensure the previous locale finishes before the next begins (on statements are non-blocking)
      var count: atomic int;
      for (loc, i) in zip(Locales, 0..) {
        on loc {
          count.waitFor(i);

          // domains of a and b that are live on this locale
          const aDom = a.localSubdomain();
          const bDom = b.localSubdomain();
          segs[here.id] = aDom.size + bDom.size;
          aMin = a[aDom.first];
          bMin = b[bDom.first];

          if loc != Locales[0] {
            // each locale determines if it needs to send values to the previous,
            //  so locale0 can skip this check (this also means indexing at [here.id - 1] is safe)

            // we've updated mins but not maxs. So mins refer to locale_i and maxs refer to locale_(i-1).
            // The previous locale needs to pull data local if it's max for one array is less
            //  than our min for the other array.
            if aMin < bMax {
              pullLocalFlag[here.id - 1] = (true, false);
              // binary search local chunk of a to find where b_min_loc_(i-1) falls,
              //  so we know how many vals to send to loc_(i-1)
              var (status, binSearchIdx) = search(a.localSlice[aDom], bMax, sorted=true);
              if (binSearchIdx == (aDom.last+1)) {
                toGiveUp = true;
              }
              // the number of elements the previous locale will need to fetch from here
              pullLocalCount[here.id - 1] = (binSearchIdx-aDom.first);
            }
            // these fetch conditions are mutally exclusive, so an else if isn't
            // strictly necessary. see logic above pullLocalCount for a pseudo proof
            if bMin < aMax {
              pullLocalFlag[here.id - 1] = (false, true);
              // binary search local chunk of b to find where a_min_loc_(i-1) falls,
              //  so we know how many vals to send to loc_(i-1)
              var (status, binSearchIdx) = search(b.localSlice[bDom], aMax, sorted=true);
              if (binSearchIdx == (bDom.last+1)) {
                toGiveUp = true;
              }
              // the number of elements the previous locale will need to fetch from here
              pullLocalCount[here.id - 1] = (binSearchIdx-bDom.first);
            }
          }
          aMax = a[aDom.last];
          bMax = b[bDom.last];

          count.add(1);
        }
        // break is not allowed in an on statement
        if toGiveUp {
          break;
        }
      }
      const totNumElemsMoved = + reduce pullLocalCount;
      const percentTransfered = (totNumElemsMoved:real / allocSize)*100:int;
      asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "Total number of elements moved to a different locale = %i".format(totNumElemsMoved));
      asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "Percent of elements moved to a different locale = %i%%".format(percentTransfered));

      if toGiveUp || (percentTransfered > percentTransferLimit) {
        // fall back to sort
        if toGiveUp {
          asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "Falling back to sort workflow since merge would need to shift an entire locale of data");
        }
        else {
          asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "Falling back to sort workflow since percent of elements moved to a different locale = %i%% exceeds percentTransferLimit = %i%%".format(percentTransfered, percentTransferLimit));
        }
        sortHelper(sortedIdx, permutedVals, idx1, idx2, val1, val2);
      }
      else {
        // we met the necessary conditions to continue with merge based workflow
        segs = (+ scan segs) - segs;
        
        // give each locale constant reference to distributed arrays that won't be edited
        // a ref to a distributed array that will be edited (but only ever one index at a time so no race conditions)
        coforall loc in Locales with (const ref a, const ref b, const ref segs, ref sortedIdx, ref permutedVals) {
          // if the chpl team implements parallel merge, that would be helpful here
          // cause we could just use their merge step instead of concatenating and doing twoArrayRadixSort
          on loc {
            // localized const copies of small arrays
            const localizedPullLocalFlag = pullLocalFlag;
            const localizedPullLocalCount = pullLocalCount;
            const aDom = a.localSubdomain();
            const bDom = b.localSubdomain();

            // the ternaries allow me to declare the next variables as const
            // info for pushing to previous locale
            const numSendBack = if loc == Locales[0] then 0 else localizedPullLocalCount[here.id - 1];
            const sendBack = if loc == Locales[0] then (false, false) else localizedPullLocalFlag[here.id - 1];
            // info for fetching from next locale
            const numPullLocal = localizedPullLocalCount[here.id];
            const pullLocal = pullLocalFlag[here.id];

            // skip any indices that were pulled to previous locale
            const aLow = if sendBack[0] then (aDom.first + numSendBack) else aDom.first;
            const bLow = if sendBack[1] then (bDom.first + numSendBack) else bDom.first;
            const aSlice = aLow..<(aDom.last + 1);
            const bSlice = bLow..<(bDom.last + 1);

            var tmp: [0..<(aDom.size + bDom.size - numSendBack + numPullLocal)] (t, t2);

            // write the local part of a (minus any skipped values) into tmp
            tmp[0..<aSlice.size] = [(key,val) in zip(a.localSlice[aSlice], aVal.localSlice[aSlice])] (key,val);
            // write the local part of b (minus any skipped values) into tmp
            tmp[aSlice.size..#bSlice.size] = [(key,val) in zip(b.localSlice[bSlice], bVal.localSlice[bSlice])] (key,val);

            // add any indices that we need to pull to this locale
            // this not a local slice, but it should (hopefully) take advantage of bulk transfer
            if pullLocal[0] {
              const pullFromASlice = (aDom.last+1)..#numPullLocal;
              tmp[(aSlice.size+bSlice.size)..#(numPullLocal)] = [(key,val) in zip(a[pullFromASlice], aVal[pullFromASlice])] (key,val);
            }
            else if pullLocal[1] {
              const pullFromBSlice = (bDom.last+1)..#numPullLocal;
              tmp[(aSlice.size+bSlice.size)..#(numPullLocal)] = [(key,val) in zip(b[pullFromBSlice], bVal[pullFromBSlice])] (key,val);
            }

            // run chpl's builtin parallel radix sort for local arrays with a comparator defined in
            // RadixSortLSD.chpl which sorts on the key
            twoArrayRadixSort(tmp, new KeysRanksComparator());

            const writeToResultSlice = (segs[here.id]+numSendBack)..#tmp.size;
            sortedIdx[writeToResultSlice] = [(key, _) in tmp] key;
            permutedVals[writeToResultSlice] = [(_, val) in tmp] val;
          }
        }
      }
    }

    proc sortHelper(ref sortedIdx: [?sD] ?t, ref permutedVals: [] ?t2, const ref idx1: [?D] t, const ref idx2: [] t, const ref val1: [] t2, const ref val2: [] t2) throws {
      const allocSize = sD.size;
      var perm = makeDistArray(allocSize, int);
      forall (s, p, sp) in zip(sortedIdx, perm, radixSortLSD(concatArrays(idx1, idx2, ordered=false))) {
        (s, p) = sp;
      }
      // concatenate the values with the same ordering as the indices
      const vals = concatArrays(val1, val2, ordered=false);
      forall (p, i) in zip(permutedVals, perm) with (var agg = newSrcAggregator(t2)) {
        agg.copy(p, vals[i]);
      }
    }

    proc combineHelper(const ref idx1: [?D] ?t, const ref idx2: [] t, const ref val1: [] ?t2, const ref val2: [] t2, doMerge = false, percentTransferLimit:int = 100) throws {
      // combine two sorted lists of indices and apply the sort permutation to their associated values
      const allocSize = idx1.size + idx2.size;
      var sortedIdx = makeDistArray(allocSize, t);
      var permutedVals = makeDistArray(allocSize, t2);
      if doMerge {
        // attempt to use merge workflow. if certain conditions are met, fall back to radixsort
        mergeHelper(sortedIdx, permutedVals, idx1, idx2, val1, val2, percentTransferLimit);
      }
      else {
        sortHelper(sortedIdx, permutedVals, idx1, idx2, val1, val2);
      }
      return (sortedIdx, permutedVals);
    }

    proc sparseSumHelper(const ref idx1: [] ?t, const ref idx2: [] t, const ref val1: [] ?t2, const ref val2: [] t2, doMerge = false, percentTransferLimit:int = 100) throws {
      const (sortedIdx, permutedVals) = combineHelper(idx1, idx2, val1, val2, doMerge, percentTransferLimit);
      const sD = sortedIdx.domain;
      var firstOccurence = makeDistArray(sD, bool);
      firstOccurence[0] = true;
      forall (f, s, i) in zip(firstOccurence, sortedIdx, sD) {
        if i > sD.low {
          // most of the time sortedIdx[i-1] should be local since we are block distributed,
          // so we only have to fetch at locale boundaries
          f = (sortedIdx[i-1] != s);
        }
      }
      const numUnique = + reduce firstOccurence;
      // we have to do a first pass through data to calculate the size of the return array
      var uIdx = makeDistArray(numUnique, t);
      var summedVals = makeDistArray(numUnique, t2);
      const retIdx = + scan firstOccurence - firstOccurence;
      forall (s, p, i, f, rIdx) in zip(sortedIdx, permutedVals, sD, firstOccurence, retIdx) with (var idxAgg = newDstAggregator(t),
                                                                                            var valAgg = newDstAggregator(t2)) {
        if f {  // skip if we are not the first occurence
          idxAgg.copy(uIdx[rIdx], s);
          if i == sD.high || sortedIdx[i+1] != s {
            valAgg.copy(summedVals[rIdx], p);
          }
          else {
            // i'd like to do aggregation but I think it's possible for remote-to-remote aggregation?
            // i.e. valAgg.copy(summedVals[rIdx], p + permutedVals[i+1]);
            // this only happens during idx collisions, so it's not the most common case
            summedVals[rIdx] = p + permutedVals[i+1];
          }
        }
      }
      return (uIdx, summedVals);
    }
}
