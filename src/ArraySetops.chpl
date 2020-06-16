/* Array set operations
 includes intersection, union, xor, and diff

 currently, only performs operations with integer arrays 
 */

module ArraySetops
{
    use ServerConfig;

    use BlockDist;
    use SymArrayDmap;

    use RadixSortLSD;
    use Unique;
    use Indexing;
    use In1d;

    use CommAggregation;

    // returns intersection of 2 arrays
    proc intersect1d(a: [?aD] int, b: [aD] int, assume_unique: string) {
      //if not unique, unique sort arrays then perform operation
      if assume_unique == "False" {
        var a1  = uniqueSortNoCounts(a);
        var b1  = uniqueSortNoCounts(b);
        return intersect1dHelper(a1, b1);
      }
      return intersect1dHelper(a,b);
    }

    // Get intersect of 2 arrays
    // first concatenates the 2 arrays, then
    // sorts arrays and removes all values that
    // only occur once
    proc intersect1dHelper(a: [?aD] ?t, b: [aD] t) {
      var aux2 = concatset(a,b);
      var aux_sort_indices = radixSortLSD_ranks(aux2);
      var aux = aux2[aux_sort_indices];

      var mask = sliceEnd(aux) == sliceStart(aux);

      var temp = sliceEnd(aux);
      var int1d = boolIndexer(temp, mask);

      return int1d;
    }

    // returns the exclusive-or of 2 arrays
    proc setxor1d(a: [?aD] int, b: [aD] int, assume_unique: string) {
      //if not unique, unique sort arrays then perform operation
      if assume_unique == "False" {
        var a1  = uniqueSortNoCounts(a);
        var b1  = uniqueSortNoCounts(b);
        return  setxor1dHelper(a1, b1);
      }
      return setxor1dHelper(a,b);
    }

    // Gets xor of 2 arrays
    // first concatenates the 2 arrays, then
    // sorts and removes all values that occur
    // more than once
    proc setxor1dHelper(a: [?aD] ?t, b: [aD] t) {
      var aux2 = concatset(a,b);
      var aux = radixSortLSD_keys(aux2);

      var sliceComp = sliceStart(aux) != sliceEnd(aux);
      var flag = concatset([true],sliceComp);
      var flag2 = concatset(flag, [true]);

      var mask = sliceStart(flag2) & sliceEnd(flag2);

      var ret = boolIndexer(aux, mask);

      return ret;
    }

    // returns the set difference of 2 arrays
    proc setdiff1d(a: [?aD] int, b: [aD] int, assume_unique: string) {
      //if not unique, unique sort arrays then perform operation
      if assume_unique == "False" {
        var a1  = uniqueSortNoCounts(a);
        var b1  = uniqueSortNoCounts(b);
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
    proc setdiff1dHelper(a: [?aD] ?t, b: [aD] t) {
        var truth = makeDistArray(a.size, bool);
        truth = in1dSort(a,b);
        truth = !truth;

        var ret = boolIndexer(a, truth);

        return ret;
    }
    
    // Gets union of 2 arrays
    // first concatenates the 2 arrays, then
    // sorts resulting array and ensures that
    // values are unique
    proc union1d(a: [?aD] int, b: [aD] int) {
      var a1  = uniqueSort(a, false);
      var b1  = uniqueSortNoCounts(b);
      var sizeA = a1.size;
      var sizeB = b1.size;

      var c = makeDistArray((sizeA + sizeB), int);

      c[{0..#sizeA}] = a;
      c[{sizeA..#sizeB}] = b;

      var ret = uniqueSortNoCounts(c);
      
      return ret;
    }
}