/* Array set operations
 includes intersection, union, xor, and diff

 currently, only performs operations with integer arrays 
 */

module ArraySetops
{
    use ServerConfig;

    use SymArrayDmapCompat;

    use RadixSortLSD;
    use Unique;
    use Indexing;
    use AryUtil;
    use In1d;
    use CommAggregation;

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

    proc sortHelper(idx1: [?D] ?t, idx2: [] t, aSize, bSize) throws {
      // all we are doing is merging two sorted lists, there has to
      // be a better way than concatenating and doing a full sort

      // eventually we want an if statement to determine if we want to merge or sort
      const allocSize = idx1.size + idx2.size;
      var sortedIdx = makeDistArray(allocSize, t);
      var perm = makeDistArray(allocSize, int);
      forall (s, p, sp) in zip(sortedIdx, perm, radixSortLSD(concatArrays(idx1, idx2))) {
        (s, p) = sp;
      }
      return (sortedIdx, perm);
    }

    proc sparseSumHelper(const ref idx1: [] ?t, const ref idx2: [] t, const ref val1: [] ?t2, const ref val2: [] t2) throws {
      const allocSize = idx1.size + idx2.size;
      const (sortedIdx, perm) = sortHelper(idx1, idx2, idx1.size, idx2.size);

      var permutedVals = makeDistArray(allocSize, t2);
      const vals = concatArrays(val1, val2);
      forall (p, i) in zip(permutedVals, perm) with (var agg = newSrcAggregator(t2)) {
        agg.copy(p, vals[i]);
      }

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
      var ret = makeDistArray(numUnique, t2);
      const retIdx = + scan firstOccurence - firstOccurence;
      forall (s, p, i, f, rIdx) in zip(sortedIdx, permutedVals, sD, firstOccurence, retIdx) with (var idxAgg = newDstAggregator(t),
                                                                                            var valAgg = newDstAggregator(t2)) {
        if f {  // skip if we are not the first occurence
          idxAgg.copy(uIdx[rIdx], s);
          if i == sD.high || sortedIdx[i+1] != s {
            valAgg.copy(ret[rIdx], p);
          }
          else {
            // i'd like to do aggregation but I think it's possible for remote-to-remote aggregation?
            // valAgg.copy(ret[rIdx], p + permutedVals[i+1]);
            ret[rIdx] = p + permutedVals[i+1];
          }
        }
      }
      return (uIdx, ret);
    }
}
