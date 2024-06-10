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
    proc sliceIndex(a: [?aD] ?t, start: int, stop: int, stride: int) throws {
      var slice: range(strides=strideKind.any);

      slice = start..(stop-1) by stride;

      var b = makeDistArray(slice.size,t);
      b = a[slice];

      return b;
    }

    // helper to get an array without the first element
    proc sliceTail(a: [?aD] ?t) throws {
      return sliceIndex(a, 1, a.size, 1);
    }

    // helper to get an array without the last element
    proc sliceHead(a: [?aD] ?t) throws {
      return sliceIndex(a, 0, a.size - 1, 1);
    }

    // return an array of all values from array a whose index corresponds to a true value in array truth
    proc boolIndexer(a: [?aD] ?t, truth: [aD] bool) throws {
        // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
        overMemLimit(numBytes(int) * truth.size);
        var iv: [truth.domain] int = (+ scan truth);
        var pop = iv[iv.size-1];
        var ret = makeDistArray(pop, t);

        forall (i, eai) in zip(a.domain, a) with (var agg = newDstAggregator(t)) {
          if (truth[i]) {
            agg.copy(ret[iv[i]-1], eai);
          }
        }
        return ret;
    }
}
