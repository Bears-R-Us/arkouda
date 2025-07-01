module SortMsg
{
    use ServerConfig;

    use Time;
    use Math only;
    use ArkoudaSortCompat only relativeComparator;
    private use DynamicSort;
    use Search only;
    use Reflection;
    use ServerErrors;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use RadixSortLSD;
    use AryUtil;
    use Logging;
    use Message;
    private use ArgSortMsg;
    use NumPyDType only whichDtype;
    use BigInteger;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const sortLogger = new Logger(logLevel, logChannel);

    /* Sort the given pdarray using Radix Sort and
       return sorted keys as a block distributed array */
    proc sort(a: [?aD] ?t): [aD] t throws {
      var sorted: [aD] t = radixSortLSD_keys(a);
      return sorted;
    }

    /* sort takes pdarray and returns a sorted copy of the array */
    @arkouda.registerCommand
    proc sort(array: [?d] ?t, alg: string, axis: int): [d] t throws
      where ((t == real) || (t == int) || (t == uint(64)))
        do return sortHelp(array, alg, axis);

    proc sortHelp(array: [?d] ?t, alg: string, axis: int): [d] t throws
      where d.rank == 1
    {
      var algorithm: SortingAlgorithm = ArgSortMsg.getSortingAlgorithm(alg);
      const itemsize = dtypeSize(whichDtype(t));
      overMemLimit(radixSortLSD_keys_memEst(d.size, itemsize));

      if algorithm == SortingAlgorithm.TwoArrayRadixSort {
        var sorted = makeDistArray(array);
        DynamicSort.dynamicTwoArrayRadixSort(sorted, comparator=myDefaultComparator);
        return sorted;
      } else {
        var sorted = radixSortLSD_keys(array);
        return sorted;
      }
    }

    proc sortHelp(array: [?d] ?t, alg: string, axis: int): [d] t throws
      where d.rank > 1
    {
      var algorithm: SortingAlgorithm = ArgSortMsg.getSortingAlgorithm(alg);
      const itemsize = dtypeSize(whichDtype(t));
      overMemLimit(radixSortLSD_keys_memEst(d.size, itemsize));

      const DD = domOffAxis(d, axis);
      var sorted = makeDistArray((...d.shape), t);

      if algorithm == SortingAlgorithm.TwoArrayRadixSort {
        for idx in DD {
          // make a copy of the array along the slice corresponding to idx
          // TODO: create a twoArrayRadixSort that operates on a slice of the array
          // in place instead of requiring the copy in/out
          var slice = makeDistArray(d.dim(axis).size, t);
          forall i in d.dim(axis) with (var perpIdx = idx) {
            perpIdx[axis] = i;
            slice[i] = array[perpIdx];
          }

          DynamicSort.dynamicTwoArrayRadixSort(slice, comparator=myDefaultComparator);

          forall i in d.dim(axis) with (var perpIdx = idx) {
            perpIdx[axis] = i;
            sorted[perpIdx] = slice[i];
          }
        }
      } else {
        // TODO: make a version of radixSortLSD_keys that does the sort on
        // slices of `e.a` directly instead of requiring a copy for each slice
        for idx in DD {
          const sliceDom = domOnAxis(d, idx, axis),
                sliced1D = removeDegenRanks(array[sliceDom], 1),
                sliceSorted = radixSortLSD_keys(sliced1D);

          forall i in sliceDom do sorted[i] = sliceSorted[i[axis]];
        }
      }

      return sorted;
    }


  proc searchSortedFast(x1: [?d1] ?t, x2: [?d2] t, side: string): [d2] int throws
    where (((d1.rank == 1) && (d2.rank == 1)) &&
            (t == int || t == real || t == uint || t == uint(8) ||
            t == bigint))
  {
    if side != "left" && side != "right" {
        throw new Error("searchSortedNew side must be a string with value \
                        'left' or 'right'.");
    }
    // This is a distributed version of searchSorted
    // We will use the local subdomain to find the boundaries for each locale
    // and then do a binary search on the local subdomain of x1 for each value
    // in x2 that lies in that locale

    // This version assumes both x1 and x2 are sorted arrays

    // Find the locale boundaries for x1
    var locBoundariesX1 = blockDist.createArray(0..Locales.size-1, (t, t));
    coforall loc in Locales do on loc {
      // Get the local subdomain of x1
      ref myLocVals = x1[x1.domain.localSubdomain()];

      // Since all arrays are always distributed across all locales in Arkouda
      // we don't need to worry about the local subdomain being empty
      // as long as the array size is greater than numLocales

      // Assign to boundaries array
      // low and high is the same as first and last since x1 is sorted
      locBoundariesX1[loc.id] = (myLocVals.first, myLocVals.last);
    }

    // Find the locale boundaries for x2, such that we know which locale
    // is responsible for which values in x2, aligning with locBoundariesX1
    // Make locBoundariesX2 a defaultDist array copied on each locale ??
    var locBoundariesX2 = blockDist.createArray(0..Locales.size-1, (int, int));
    coforall loc in Locales do on loc {
      // Get the low and high values for the local subdomain of x1
      const (myLow, myHigh) = locBoundariesX1[loc.id];
      const isFirstLocale = here.id == 0;
      const isLastLocale = here.id == Locales.size - 1;

      const prevLocId = if isFirstLocale then 0 else here.id - 1; // previous locale id, won't be used for locale 0
      const (_, prevHigh) = if isFirstLocale then (0:t, myLow) else locBoundariesX1[prevLocId];
      const nextLocId = if isLastLocale then Locales.size - 1 else here.id + 1 ; // next locale id, won't be used for last locale
      const (nextLow, _) = if isLastLocale then (myHigh, 0:t) else locBoundariesX1[nextLocId];

      // Use binary search to find boundaries efficiently
      var myFirst = -1;
      var myLast = -1;

      // Find first index where x2[i] >= myLow
      const (_, eqOrGtMyLow) = if myLow == prevHigh then
        Search.binarySearch(x2, myLow, new leftCmp())
      else
        Search.binarySearch(x2, prevHigh, new rightCmp())
      ;
      // Special case for the 0th Locale, where myLow really doesn't matter,
      // we need to include all elements < myHigh
      if isFirstLocale && eqOrGtMyLow > x2.domain.high {
        myFirst = 0; // We own all elements in x2
      } else if eqOrGtMyLow <= x2.domain.high { // eqOrGtMyLow can be out of bounds if myLow is > all elements in x2 for other locales

        const eqOrGtElem = x2[eqOrGtMyLow]; // this is the element at the index we found

        // If this value is > myHigh, we can skip the rest since it's not in our range
        if isLastLocale && eqOrGtElem > myHigh { // special case for last locale, where myHigh really doesn't matter,
          myFirst = eqOrGtMyLow; // We own this value, so we can set myFirst to eqOrGtMyLow
        } else if !isFirstLocale && eqOrGtElem == myLow && eqOrGtElem == prevHigh
          /*&& prevHigh == myLow (implied)*/ && side == "left" {

          // If we are on any locale other than the first one,
          // AND eqOrGtElem is equal to myLow,
          // AND eqOrGtElem is equal to the previous locale's high,
          // AND side is "left", then the previous locale owns this value

          // The next value we could possibly own is the first x2 value that is greater than myLow
          // Find first index where x2[i] > myLow
          const (_, gtMyLow) = Search.binarySearch(x2, prevHigh, new rightCmp());
          // gtMyLow can be out of bounds if myLow is >= all elements in x2
          if gtMyLow <= x2.domain.high {
            const gtElem = x2[gtMyLow]; // this is the element at the index we found
            // If this value is > myHigh, we can skip the rest since it's not in our range (except for last locale)
            // Or if gtElem is same as eqOrGtElem, then we don't own this either
            if (isLastLocale || gtElem <= myHigh) && gtElem != eqOrGtElem {
            // We own this value, so we can set myFirst to gtMyLow
            myFirst = gtMyLow;
            }
          }
        } else {
          // We own this value, so we can set myFirst to eqOrGtMyLow
          myFirst = eqOrGtMyLow;
        }
      }

      // Now find myLast using binary search instead of linear search
      if myFirst != -1 {
        // Find the last index where x2[i] <= myHigh
        const (_, eqOrLeMyHighPlusOne) = Search.binarySearch(x2, myHigh, new rightCmp()); // right cmp is correct here always, since we want the last index
        // The element and index we have is 1 after the last element that is <= myHigh
        // so we go back one index to get the last element that is <= myHigh
        const eqOrLeMyHigh = eqOrLeMyHighPlusOne - 1;
        // eqOrLeMyHigh can also be out of bounds if myHigh is < all elements in x2
        // Special case for the last Locale, where myHigh really doesn't matter,
        // we need to include all elements < myHigh
        if isLastLocale {
          myLast = x2.domain.high;
        } else if eqOrLeMyHigh >= x2.domain.low {
          const eqOrLeElem = x2[eqOrLeMyHigh]; // this is the element at the index we found

          // If we are any locale other than the last one,
          // AND eqOrLeElem is equal to myHigh,
          // AND eqOrLeElem is equal to the next locale's low,
          // AND side is "right", then the next locale owns this value
          if !isLastLocale && eqOrLeElem == myHigh && eqOrLeElem == nextLow
            /*&& nextLow == myHigh (implied)*/ && side == "right" {
            // Find last index where x2[i] < myHigh
            const (_, ltMyHighPlusOne) = Search.binarySearch(x2, myHigh, new leftCmp());
            // The element and index we have is 1 after the last element that is < myHigh
            // so we go back one index to get the last element that is < myHigh
            const ltMyHigh = ltMyHighPlusOne - 1;
            // ltMyHigh can be out of bounds if myHigh is <= all elements in x2
            /// TODOOOOOOO special case last locale ??? (not needed since this will never trigger because of the blanket last locale if statement setting myLast above)
            if ltMyHigh >= x2.domain.low {
              const ltElem = x2[ltMyHigh]; // this is the element at the index we found
              // If this value is < prevHigh, we can skip the rest since it's not in our range
              // Or if ltElem is same as eqOrLeElem, then we don't own this either
              if ltElem >= prevHigh && ltElem != eqOrLeElem {
                // We own this value, so we can set myLast to ltMyHigh
                myLast = ltMyHigh;
              }
            }
          } else {
            // We own this value, so we can set myLast to eqOrLeMyHigh
            myLast = eqOrLeMyHigh;
          }
        }
      }

      // Handle empty ranges
      if myFirst == -1 {
        myFirst = 0;
        myLast = -1; // This creates an empty range [0..-1]
      }

      // Assign the boundaries for this locale
      locBoundariesX2[loc.id] = (myFirst, myLast);
    }
    // Now we have the boundaries for each locale, we can do a binary search on x1
    // for each value in my chunk of x2
    var ret = x2.domain.tryCreateArray(int); // change to makeDistArray
    coforall loc in Locales do on loc {
      // Get the local subdomain of x1
      const myLocalSubdomainX1 = x1.domain.localSubdomain();
      ref myLocX1 = x1[myLocalSubdomainX1];

      // Get the boundaries for this locale
      const (myFirst, myLast) = locBoundariesX2[loc.id];

      // Create a local copy subdomain of x2 that this locale is responsible for
      var myLocX2 : [myFirst..myLast] t;
      // This seems risky. What if this slice is too big to fit on a single node?
      // Ex: all the elements in x2 are smaller than x1[0], so the entirety of x2
      // is assigned to locale 0? I guess this is also a load balancing issue...
      myLocX2[myFirst..myLast] = x2[myFirst..myLast];

      select side {
        when "left" do doSearch(myLocX1, myLocX2, new leftCmp());
        when "right" do doSearch(myLocX1, myLocX2, new rightCmp());
        otherwise do halt("unreachable");
       }
    }

    proc doSearch(const ref a1: [] t, const ref a2: [?d] t, cmp) {
      var localret : [d] int;
      forall idx in d {
        const (_, i) = Search.binarySearch(a1, a2[idx], cmp);
        // ret[idx] = i;
        localret[idx] = i;
      }
      ret[d] = localret[d];
    }

    return ret;
  }


    // https://data-apis.org/array-api/latest/API_specification/generated/array_api.searchsorted.html#array_api.searchsorted
    @arkouda.registerCommand
    proc searchSorted(x1: [?d1] ?t, x2: [?d2] t, side: string, x2Sorted : bool = false): [d2] int throws
      where ((d1.rank == 1) &&
             (t == int || t == real || t == uint || t == uint(8) || t == bigint))
    {
      if side != "left" && side != "right" {
          throw new Error("searchSorted side must be a string with value 'left' or 'right'.");
      }

      if (x2.rank == 1) && x2Sorted && x1.size >= numLocales {
        // If x2 is already sorted, we can use the fast version
        param msg = "Fast searchSorted path taken";
        sortLogger.info(getModuleName(),getRoutineName(),getLineNumber(), msg);
        return searchSortedFast(x1, x2, side);
      }
      return searchSortedSlow(x1, x2, side);
    }


    proc searchSortedSlow(x1: [?d1] ?t, x2: [?d2] t, side: string): [d2] int throws
      where ((d1.rank == 1) &&
             (t == int || t == real || t == uint || t == uint(8) || t == bigint))
    {

      var ret = makeDistArray((...x2.shape), int);

      proc doSearch(const ref a1: [] t, const ref a2: [?d] t, cmp) {
        forall idx in ret.domain {
          const (_, i) = Search.binarySearch(a1, a2[idx], cmp);
          ret[idx] = i;
        }
      }

      select side {
        when "left" do doSearch(x1, x2, new leftCmp());
        when "right" do doSearch(x1, x2, new rightCmp());
        otherwise do halt("unreachable");
      }

      return ret;
    }

    record leftCmp: relativeComparator {
      proc compare(a: ?t, b: t): int
        where (t == int || t == real || t == uint || t == uint(8) || t == bigint)
      {
        if a <= b then return -1;
        else return 1;
      }
    }

    record rightCmp: relativeComparator {
      proc compare(a: ?t, b: t): int
        where (t == int || t == real || t == uint || t == uint(8) || t == bigint)
      {
        if a < b then return -1;
        else return 1;
      }
    }
}// end module SortMsg
