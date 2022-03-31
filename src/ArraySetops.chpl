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
    use AryUtil;
    use In1d;

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
    proc setdiff1d(a: [] ?t, b: [] t, assume_unique: bool) throws {
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
    proc setdiff1dHelper(a: [] ?t, b: [] t) throws {
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
}
