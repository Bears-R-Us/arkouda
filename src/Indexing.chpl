/* Helper functions for indexing a chapel array
   includes slicing, indexing by boolean ararys, and concatenation
 */

module Indexing {
    use ServerConfig;
    use ServerErrorStrings;

    use Reflection only;

    use MultiTypeSymEntry;
    use MultiTypeSymbolTable;

    use CommAggregation;
    
    // Return a slice of array `a` from `start` to `stop` by `stride`
    proc sliceIndex(a: [?aD] ?t, start: int, stop: int, stride: int) {
      var slice: range(stridable=true);
      
      slice = start..(stop-1) by stride;

      var b = makeDistArray(slice.size,t);
      b = a[slice];

      return b;
    }

    // helper to get an array without the first element
    proc sliceTail(a: [?aD] ?t) {
      return sliceIndex(a, 1, a.size, 1);
    }

    // helper to get an array without the last element
    proc sliceHead(a: [?aD] ?t) {
      return sliceIndex(a, 0, a.size - 1, 1);
    }

    // return an array of all values from array a whose index corresponds to a true value in array truth
    proc boolIndexer(a: [?aD] ?t, truth: [aD] bool) {
        var iv: [truth.domain] int = (+ scan truth);
        var pop = iv[iv.size-1];
        var ret = makeDistArray(pop, int);

        forall (i, eai) in zip(a.domain, a) with (var agg = newDstAggregator(int)) {
          if (truth[i]) {
            agg.copy(ret[iv[i]-1], eai);
          }
        }
        return ret;
    }

    // concatenate 2 distributed arrays and return the result
    proc concatset(a: [?aD] ?t, b: [?bD] t) {
      var ret = makeDistArray((a.size + b.size), t);
      
      ret[{0..#a.size}] = a;
      ret[{a.size..#b.size}] = b;

      return ret;
    }
}