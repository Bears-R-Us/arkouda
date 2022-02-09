module In1d
{
    use ServerConfig;
    use Unique;
    use AryUtil;
    use CommAggregation;
    use RadixSortLSD;
    use Reflection;

    /* Put ar2 into an associative domain of int, per locale. 
    Creates truth (boolean array) from the domain of ar1.
    ar1 and truth are distributed on the same locales.
    ar2 is copied to a set (associative domain) in each locale.
    set membership of ar1 to ar2 is checked on each locale by iterating over 
    the local subdomains of ar1, and populating the local subdomains of truth 
    with the result of the membership test.

    :arg ar1: array to broadcast in parallel over ar2
    :type ar1: [] int
    
    :arg ar2: array to be broadcast over in parallel
    :type ar2: [] int
    
    :returns truth: the distributed boolean array containing the result of ar1 being broadcast over ar2
    :type truth: [] bool
    */
    proc in1dAr2PerLocAssoc(ar1: [?aD1] ?t, ar2: [?aD2] t) {
        var truth: [aD1] bool;
        
        coforall loc in Locales {
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

    /* For each value in the first array, check membership in the second array. This 
       implementation uses a sort, which is best when the second array is large because 
       it scales well in both time and memory.

       :arg ar1: array to broadcast in parallel over ar2
       :type ar1: [] int
    
       :arg ar2: array to be broadcast over in parallel
       :type ar2: [] int
    
       :returns truth: the distributed boolean array containing the result of ar1 being broadcast over ar2
       :type truth: [] bool
     */
    proc in1dSort(ar1: [?aD1] ?t, ar2: [?aD2] t) throws  {
        /* General strategy: unique both arrays, find the intersecting values, 
           then map back to the original domain of ar1.
         */
        // Need the inverse index to map back from unique domain to original domain later
        var (u1, c1, inv) = uniqueSortWithInverse(ar1);
        var (u2, c2) = uniqueSort(ar2);
        // Concatenate the two unique arrays
        const D = makeDistDom(u1.size + u2.size);
        var ar: [D] t;
        /* ar[{0..#u1.size}] = u1; */
        ar[D.interior(-u1.size)] = u1;
        /* ar[{u1.size..#u2.size}] = u2; */
        ar[D.interior(u2.size)] = u2;
        // Sort unique arrays together to find duplicates
        var sar: [D] t;
        var order: [D] int;
        forall (s, o, so) in zip(sar, order, radixSortLSD(ar)) {
          (s, o) = so;
        }
        // Duplicates correspond to values in both arrays
        var flag: [D] bool;
        flag[D.interior(-(D.size-1))] = (sar[D.interior(D.size-1)] == sar[D.interior(-(D.size-1))]);
        // Get the indices of values from u1 that are also in u2
        // Because sort is stable, original index of left duplicate will always be in u1
        var ret: [D] bool;
        forall (o, f) in zip(order, flag) with (var agg = newDstAggregator(bool)) {
            agg.copy(ret[o], f);
        }
        // Use the inverse index to map from u1 domain to ar1 domain
        var truth: [aD1] bool;
        forall (t, idx) in zip(truth, inv) with (var agg = newSrcAggregator(bool)) {
            agg.copy(t, ret[idx]);
        }
        return truth;
    }
}
