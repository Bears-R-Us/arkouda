/* Array set operations
 includes intersection, union, xor, and diff

 currently, only performs operations with integer arrays 
 */

module ArraySetops
{
    use ServerConfig;

    use SymArrayDmap;

    use RadixSortLSD;
    use Unique;
    use Indexing;
    use In1d;


    /*
    Small bound const. Brute force in1d implementation recommended.
    */
    private config const sBound = 2**4; 

    /*
    Medium bound const. Per locale associative domain in1d implementation recommended.
    */
    private config const mBound = 2**25; 

    // returns intersection of 2 arrays
    proc intersect1d(a: [] int, b: [] int, assume_unique: bool) {
      //if not unique, unique sort arrays then perform operation
      if (!assume_unique) {
        var a1  = uniqueSort(a, false);
        var b1  = uniqueSort(b, false);
        return intersect1dHelper(a1, b1);
      }
      return intersect1dHelper(a,b);
    }

    // Get intersect of 2 arrays
    // first concatenates the 2 arrays, then
    // sorts arrays and removes all values that
    // only occur once
    proc intersect1dHelper(a: [] ?t, b: [] t) {
      var aux = radixSortLSD_keys(concatset(a,b));

      var head = sliceEnd(aux);
      var mask = head == tail(aux);
      
      var int1d = boolIndexer(head, mask);

      return int1d;
    }

    // returns the exclusive-or of 2 arrays
    proc setxor1d(a: [] int, b: [] int, assume_unique: bool) {
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
    proc setxor1dHelper(a: [] ?t, b: [] t) {
      var aux = radixSortLSD_keys(concatset(a,b));

      var sliceComp = tail(aux) != sliceEnd(aux);
      
      // Concatenate a `true` onto each end of the array
      var flag = makeDistArray((sliceComp.size + 2), bool);
      
      flag[0] = true;
      flag[{1..#(sliceComp.size)}] = sliceComp;
      flag[sliceComp.size + 1] = true;

      var mask = tail(flag) & sliceEnd(flag);

      var ret = boolIndexer(aux, mask);

      return ret;
    }

    // returns the set difference of 2 arrays
    proc setdiff1d(a: [] int, b: [] int, assume_unique: bool) {
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
    proc setdiff1dHelper(a: [] ?t, b: [] t) {
        var truth = makeDistArray(a.size, bool);

        // based on size of array, determine which method to use 
        if (b.size <= sBound) then truth = in1dGlobalAr2Bcast(a, b);
        else if (b.size <= mBound) then truth = in1dAr2PerLocAssoc(a, b);
        else truth = in1dSort(a,b);
        
        truth = !truth;

        var ret = boolIndexer(a, truth);

        return ret;
    }
    
    // Gets union of 2 arrays
    // first concatenates the 2 arrays, then
    // sorts resulting array and ensures that
    // values are unique
    proc union1d(a: [] int, b: [] int) {
      var a1  = uniqueSort(a, false);
      var b1  = uniqueSort(b, false);

      var aux = concatset(a1, b1);

      var ret = uniqueSort(aux, false);
      
      return ret;
    }
}