module SegArraySetops {
  use ServerConfig;
  use ArraySetops;
  use CommAggregation;
  use AryUtil;
  use MultiTypeSymEntry;

  proc union_sa(segments1: borrowed SymEntry(int), values1: borrowed SymEntry(?t), lens1: [] int, segments2: borrowed SymEntry(int), values2: borrowed SymEntry(t), lens2: [] int) throws {
    var union_lens: [segments1.aD] int;

    // Compute lengths of the segments resulting from each union
    forall (idx, s1, l1, s2, l2, ul) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, union_lens) with (var agg = newDstAggregator(int)){
      // TODO - update to use lowLevelLocalizingSlice 
      if (l1 == 0) {
        agg.copy(ul, l2);
      }
      else if (l2 == 0) {
        agg.copy(ul, l1);
      }
      else{
        var u = union1d(values1.a[s1..#l1], values2.a[s2..#l2]);
        agg.copy(ul, u.size);
      }
    }

    const union_segs = (+ scan union_lens) - union_lens;
    var union_vals = makeDistArray((+ reduce union_lens), t);

    // Compute the union and add values to the corresponding indexes in values
    forall (idx, s1, l1, s2, l2, us, ul) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, union_segs, union_lens) with (var agg = newDstAggregator(t)){
      // TODO - update to use lowLevelLocalizingSlice 
      if (l1 == 0){
        var u = new lowLevelLocalizingSlice(values2.a, s2..#l2);
        for i in (0..#l2) {
          agg.copy(union_vals[i+us], u.ptr[i]);
        }
      } else if (l2 == 0) {
        var u = new lowLevelLocalizingSlice(values1.a, s1..#l1);
        for i in (0..#l1) {
          agg.copy(union_vals[i+us], u.ptr[i]);
        }
      } else {
        var u = new lowLevelLocalizingSlice(union1d(values1.a[s1..#l1], values2.a[s2..#l2]), 0..#ul);
        for i in (0..#ul){
          agg.copy(union_vals[i+us], u.ptr[i]);
        }
      }
    }

    return (union_segs, union_vals);
  }

  proc intersect(segments1: borrowed SymEntry(int), values1: borrowed SymEntry(?t), lens1: [] int, segments2: borrowed SymEntry(int), values2: borrowed SymEntry(t), lens2: [] int, isUnique: bool) throws {
    var intx_lens: [segments1.aD] int;

    // Compute lengths of the segments resulting from each union
    forall (idx, s1, l1, s2, l2, il) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, intx_lens) with (var agg = newDstAggregator(int)){
      // TODO - update to use lowLevelLocalizingSlice 
      // if either segment is empty, intersection is empty
      if (l1 == 0 || l2 == 0) {
        agg.copy(il, 0);
      } else {
        var intx = intersect1d(values1.a[s1..#l1], values2.a[s2..#l2], isUnique);
        agg.copy(il, intx.size);
      }
    }

    const intx_segs = (+ scan intx_lens) - intx_lens;
    var intx_vals = makeDistArray((+ reduce intx_lens), t);

    // Compute the intersection and add values to the corresponding indexes in values
    forall (idx, s1, l1, s2, l2, is, il) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, intx_segs, intx_lens) with (var agg = newDstAggregator(t)){
      // TODO - update to use lowLevelLocalizingSlice 
      if (il > 0){
        var intx = new lowLevelLocalizingSlice(intersect1d(values1.a[s1..#l1], values2.a[s2..#l2], isUnique), 0..#il);
        for i in (0..#il){
            agg.copy(intx_vals[i+is], intx.ptr[i]);
        }
      }
    }

    return (intx_segs, intx_vals);
  }

  proc setxor(segments1: borrowed SymEntry(int), values1: borrowed SymEntry(?t), lens1: [] int, segments2: borrowed SymEntry(int), values2: borrowed SymEntry(t), lens2: [] int, isUnique: bool) throws {
    var xor_lens: [segments1.aD] int;

    // Compute lengths of the segments resulting from each union
    forall (idx, s1, l1, s2, l2, xl) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, xor_lens) with (var agg = newDstAggregator(int)){
      // TODO - update to use lowLevelLocalizingSlice 
      var xor = setxor1d(values1.a[s1..#l1], values2.a[s2..#l2], isUnique);
      agg.copy(xl, xor.size);
    }

    const xor_segs = (+ scan xor_lens) - xor_lens;
    var xor_vals = makeDistArray((+ reduce xor_lens), t);

    // Compute the setxor and add values to the corresponding indexes in values
    forall (idx, s1, l1, s2, l2, xs, xl) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, xor_segs, xor_lens) with (var agg = newDstAggregator(t)){
      // TODO - update to use lowLevelLocalizingSlice 
      var xor = new lowLevelLocalizingSlice(setxor1d(values1.a[s1..#l1], values2.a[s2..#l2], isUnique), 0..#xl);
      for i in (0..#xl){
        agg.copy(xor_vals[i+xs], xor.ptr[i]);
      }
    }

    return (xor_segs, xor_vals);
  }

  proc setdiff(segments1: borrowed SymEntry(int), values1: borrowed SymEntry(?t), lens1: [] int, segments2: borrowed SymEntry(int), values2: borrowed SymEntry(t), lens2: [] int, isUnique: bool) throws {
    var diff_lens: [segments1.aD] int;

    // Compute lengths of the segments resulting from each union
    forall (idx, s1, l1, s2, l2, dl) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, diff_lens) with (var agg = newDstAggregator(int)){
      // TODO - update to use lowLevelLocalizingSlice 
      if (l1 == 0 || l2 == 0){
        agg.copy(dl, l1);
      } else {
        var d = setdiff1d(values1.a[s1..#l1], values2.a[s2..#l2], isUnique);
        agg.copy(dl, d.size);
      }
    }

    const diff_segs = (+ scan diff_lens) - diff_lens;
    var diff_vals = makeDistArray((+ reduce diff_lens), t);

    // Compute the difference and add values to the corresponding indexes in values
    forall (idx, s1, l1, s2, l2, ds, dl) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, diff_segs, diff_lens) with (var agg = newDstAggregator(t)){
      // TODO - update to use lowLevelLocalizingSlice 
      if (l1 == 0 || l2 == 0){
        var d = new lowLevelLocalizingSlice(values1.a, s1..#l1);
        for i in (0..#l1) {
          agg.copy(diff_vals[i + ds], d.ptr[i]);
        }
      } else {
        var d = new lowLevelLocalizingSlice(setdiff1d(values1.a[s1..#l1], values2.a[s2..#l2], isUnique), 0..#dl);
        for i in (0..#dl){
          agg.copy(diff_vals[i+ds], d.ptr[i]);
        }
      }
    }

    return (diff_segs, diff_vals);
  }
    
}