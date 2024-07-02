module In1d
{
    use ServerConfig;
    use Unique;
    use AryUtil;
    use CommAggregation;
    use RadixSortLSD;
    use Reflection;
    use List;
    use SegmentedString;

    /* Threshold for choosing between in1d implementation strategies */
    private config const threshold = 2**23;

    /* For each value in the first array, check membership in the second array.

       :arg ar1: array to broadcast in parallel over ar2
       :type ar1: [] int

       :arg ar2: array to be broadcast over in parallel
       :type ar2: [] int

       :arg invert: should the result be inverted (not in1d)
       :type invert: bool

       :returns truth: the distributed boolean array containing the result of ar1 being broadcast over ar2
       :type truth: [] bool
     */
    proc in1d(ar1: [?aD1] ?t, ref ar2: [?aD2] t, invert: bool = false): [aD1] bool throws {
        var truth = if ar2.size <= threshold then in1dAr2PerLocAssoc(ar1, ar2)
                                             else in1dSort(ar1, ar2);
        if invert then truth = !truth;
        return truth;
    }

    /* in1d that uses a per-locale set/associative-domain. Each locale will
     * localize ar2 and put it in the set, so only appropriate in terms of
     * size and space when ar2 is "small".
     */
    proc in1dAr2PerLocAssoc(ar1: [?aD1] ?t, ref ar2: [?aD2] t) throws {
        var truth = makeDistArray(aD1, bool);
        
        coforall loc in Locales with (ref truth, ref ar2) {
            on loc {
                var ar2Set: domain(t, parSafe=false); // create a set to hold ar2, parSafe modification is OFF
                ar2Set.requestCapacity(ar2.size); // request a capacity for the initial set

                for loc in offset(0..<numLocales) {
                    var lD = ar2.localSubdomain(Locales[loc]);
                    var slice = new lowLevelLocalizingSlice(ar2, lD.low..lD.high);
                    // serially add all elements of ar2 to ar2Set
                    for i in 0..<lD.size { ar2Set += slice.ptr[i]; }
                }

                // in parallel check all elements of ar1 to see if ar2Set contains them
                [i in truth.localSubdomain()] truth[i] = ar2Set.contains(ar1[i]);
            }
        }
        return truth;
    }

    /* in1d that uses a sorting strategy. At a high level it uniques both
     * arrays, finds the intersecting values, then maps back to the original
     * domain of ar1. Scales well with time/size, but sort has non-trivial
     * overhead so typically used when ar2 is "large".
     */
    proc in1dSort(ar1: [?aD1] ?t, ar2: [?aD2] t) throws {
        // Need the inverse index to map back from unique domain to original domain later
        var (u1, _, inv) = uniqueSortWithInverse(ar1);
        var u2 = uniqueSort(ar2, needCounts=false);
        // Concatenate the two unique arrays
        const ar = concatArrays(u1, u2);
        const D = ar.domain;
        // Sort unique arrays together to find duplicates
        var sar = makeDistArray(D, t);
        var order = makeDistArray(D, int);
        forall (s, o, so) in zip(sar, order, radixSortLSD(ar)) {
            (s, o) = so;
        }
        // Duplicates correspond to values in both arrays
        var flag = makeDistArray(D, bool);
        forall i in D[D.low..<D.high] with (var agg = newDstAggregator(bool)) {
            agg.copy(flag[order[i]], sar[i+1] == sar[i]);
        }
        // Use the inverse index to map from u1 domain to ar1 domain
        var truth = makeDistArray(aD1, bool);
        forall (t, idx) in zip(truth, inv) with (var agg = newSrcAggregator(bool)) {
            agg.copy(t, flag[idx]);
        }
        return truth;
    }
}
