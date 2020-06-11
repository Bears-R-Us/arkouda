module NewSetxor1d
{
    use ServerConfig;

    use BlockDist;
    use SymArrayDmap;

    use RadixSortLSD;
    use Unique;
    use NewUnion1d;
    use Indexing;
    use In1d;

    use CommAggregation;
    
    proc newSetxor1d(a: [?aD] int, b: [aD] int, assume_unique: string) {
      if assume_unique == "False" {
        var (a1, _)  = uniqueSort(a);
        var (b1, _)  = uniqueSort(b);
      }

      var aux = concatset(a,b);

      var aux_sort_indides = radixSortLSD_ranks(aux);
      aux = aux[aux_sort_indides];

      var flag = true;//concatset([true],[(sliceIndex(aux,1,aux.size,1) != sliceIndex(aux,0,aux.size,1))]);
      
      return a;//aux[flag[1:] & flag[:-1]];
    }

    proc concatset(a: [?aD] ?t, b: [?bD] t) {
      var sizeA = a.size;
      var sizeB = b.size;
      select t {
          when int {
            var ret = makeDistArray((sizeA + sizeB), int);
            ret[{0..#sizeA}] = a;
            ret[{sizeA..#sizeB}] = b;

            return ret;
          }
          when bool {
            var ret = makeDistArray((sizeA + sizeB), bool);
            ret[{0..#sizeA}] = a;
            ret[{sizeA..#sizeB}] = b;

            return ret;
          }
       }
    }

    proc newSetdiff1d(a: [?aD] int, b: [?bD] int, assume_unique: string) {
      if assume_unique == "False" {
        var (a1, _)  = uniqueSort(a);
        var (b1, _)  = uniqueSort(b);
        var truth = makeDistArray(a1.size, bool);
        truth = in1dGlobalAr2Bcast(a1,b1);
        truth = !truth;

        var iv: [truth.domain] int = (+ scan truth);
        var pop = iv[iv.size-1];
        var ret = makeDistArray(pop, int);

        forall (i, eai) in zip(a1.domain, a1) with (var agg = newDstAggregator(int)) {
          if (truth[i]) {
            agg.copy(ret[iv[i]-1], eai);
          }
        }

        return ret;
      }
      else {
        var truth = makeDistArray(a.size, bool);
        truth = in1dGlobalAr2Bcast(a,b);
        truth = !truth;

        var iv: [truth.domain] int = (+ scan truth);
        var pop = iv[iv.size-1];
        var ret = makeDistArray(pop, int);

        forall (i, eai) in zip(aD, a) with (var agg = newDstAggregator(int)) {
          if (truth[i]) {
            agg.copy(ret[iv[i]-1], eai);
          }
        }

        return ret;
      }
    }
}