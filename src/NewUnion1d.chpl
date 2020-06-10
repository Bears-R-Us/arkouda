module NewUnion1d
{
    use ServerConfig;
    
    use BlockDist;
    use SymArrayDmap;
    
    use RadixSortLSD;
    use Unique;

    proc newUnion1d(a: [?aD] int, b: [aD] int) {
      var (a1, _)  = uniqueSort(a);
      var (b1, _)  = uniqueSort(b);
      var sizeA = a1.size;
      var sizeB = b1.size;

      var c = makeDistArray((sizeA + sizeB), int);

      c[{0..#sizeA}] = a;
      c[{sizeA..#sizeB}] = b;

      var (ret, _) = uniqueSort(c);
      
      return ret;
    }
}