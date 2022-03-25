/* Array set operations
 includes intersection, union, xor, and diff

 currently, only performs operations with integer arrays 
 */

module ArraySetops
{
    use ServerConfig;

    use SymArrayDmap;
    use MultiTypeSymEntry;

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

    proc intersect1d_multi(segments1: borrowed SymEntry(int), values1: borrowed SymEntry(?t), lens1: [] int, segments2: borrowed SymEntry(int), values2: borrowed SymEntry(t), lens2: [] int, isUnique: bool) throws {
      var intx_lens: [segments1.aD] int;

      // Compute lengths of the segments resulting from each union
      forall (i, s1, l1, s2, l2, il) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, intx_lens) with (var agg = newDstAggregator(int)){
        // TODO - update to use lowLevelLocalizingSlice 
        var intx = intersect1d(values1.a[s1..#l1], values2.a[s2..#l2], isUnique);
        agg.copy(il, intx.size);
      }

      const intx_segs = (+ scan intx_lens) - intx_lens;
      var intx_vals = makeDistArray((+ reduce intx_lens), t);

      // Compute the union and add values to the corresponding indexes in values
      forall (i, s1, l1, s2, l2, is, il) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, intx_segs, intx_lens) with (var agg = newDstAggregator(t)){
        // TODO - update to use lowLevelLocalizingSlice 
        var intx = intersect1d(values1.a[s1..#l1], values2.a[s2..#l2], isUnique);
        for i in (0..#il){
          agg.copy(intx_vals[i+is], intx[i]);
        }
      }

      return (intx_segs, intx_vals);
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

    proc setxor1d_multi(segments1: borrowed SymEntry(int), values1: borrowed SymEntry(?t), lens1: [] int, segments2: borrowed SymEntry(int), values2: borrowed SymEntry(t), lens2: [] int, isUnique: bool) throws {
      var xor_lens: [segments1.aD] int;

      // Compute lengths of the segments resulting from each union
      forall (i, s1, l1, s2, l2, xl) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, xor_lens) with (var agg = newDstAggregator(int)){
        // TODO - update to use lowLevelLocalizingSlice 
        var xor = setxor1d(values1.a[s1..#l1], values2.a[s2..#l2], isUnique);
        agg.copy(xl, xor.size);
      }

      const xor_segs = (+ scan xor_lens) - xor_lens;
      var xor_vals = makeDistArray((+ reduce xor_lens), t);

      // Compute the union and add values to the corresponding indexes in values
      forall (i, s1, l1, s2, l2, xs, xl) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, xor_segs, xor_lens) with (var agg = newDstAggregator(t)){
        // TODO - update to use lowLevelLocalizingSlice 
        var xor = setxor1d(values1.a[s1..#l1], values2.a[s2..#l2], isUnique);
        for i in (0..#xl){
          agg.copy(xor_vals[i+xs], xor[i]);
        }
      }

      return (xor_segs, xor_vals);
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

    proc setdiff1d_multi(segments1: borrowed SymEntry(int), values1: borrowed SymEntry(?t), lens1: [] int, segments2: borrowed SymEntry(int), values2: borrowed SymEntry(t), lens2: [] int, isUnique: bool) throws {
      var diff_lens: [segments1.aD] int;

      // Compute lengths of the segments resulting from each union
      forall (i, s1, l1, s2, l2, dl) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, diff_lens) with (var agg = newDstAggregator(int)){
        // TODO - update to use lowLevelLocalizingSlice 
        var d = setdiff1d(values1.a[s1..#l1], values2.a[s2..#l2], isUnique);
        agg.copy(dl, d.size);
      }

      const diff_segs = (+ scan diff_lens) - diff_lens;
      var diff_vals = makeDistArray((+ reduce diff_lens), t);

      // Compute the union and add values to the corresponding indexes in values
      forall (i, s1, l1, s2, l2, ds, dl) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, diff_segs, diff_lens) with (var agg = newDstAggregator(t)){
        // TODO - update to use lowLevelLocalizingSlice 
        var d = setdiff1d(values1.a[s1..#l1], values2.a[s2..#l2], isUnique);
        for i in (0..#dl){
          agg.copy(diff_vals[i+ds], d[i]);
        }
      }

      return (diff_segs, diff_vals);
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

    proc union1d_multi(segments1: borrowed SymEntry(int), values1: borrowed SymEntry(?t), lens1: [] int, segments2: borrowed SymEntry(int), values2: borrowed SymEntry(t), lens2: [] int) throws {
      var union_lens: [segments1.aD] int;

      // Compute lengths of the segments resulting from each union
      forall (i, s1, l1, s2, l2, ul) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, union_lens) with (var agg = newDstAggregator(int)){
        // TODO - update to use lowLevelLocalizingSlice 
        var u = union1d(values1.a[s1..#l1], values2.a[s2..#l2]);
        agg.copy(ul, u.size);
      }

      const union_segs = (+ scan union_lens) - union_lens;
      var union_vals = makeDistArray((+ reduce union_lens), t);

      // Compute the union and add values to the corresponding indexes in values
      forall (i, s1, l1, s2, l2, us, ul) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, union_segs, union_lens) with (var agg = newDstAggregator(t)){
        // TODO - update to use lowLevelLocalizingSlice 
        var u = union1d(values1.a[s1..#l1], values2.a[s2..#l2]);
        for i in (0..#ul){
          agg.copy(union_vals[i+us], u[i]);
        }
      }

      return (union_segs, union_vals);
    }
}
