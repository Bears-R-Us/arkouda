module Indexing {
    use ServerConfig;
    use ServerErrorStrings;

    use Reflection only;

    use MultiTypeSymEntry;
    use MultiTypeSymbolTable;

    use CommAggregation;

    proc sliceIndex(a: [?aD] ?t, start: int, stop: int, stride: int) {
      var slice: range(stridable=true);

      // convert python slice to chapel slice
      // backwards iteration with negative stride
      if  (start > stop) & (stride < 0) {slice = (stop+1)..start by stride;}
      // forward iteration with positive stride
      else if (start <= stop) & (stride > 0) {slice = start..(stop-1) by stride;}
      // BAD FORM start < stop and stride is negative
      else {slice = 1..0;}

      var b = makeDistArray(slice.size,t);
      b = a[slice];

      return b;
    }

    proc testSlice() {
      return sliceIndex([1,2,3], 0, 1, 1);
    }
}