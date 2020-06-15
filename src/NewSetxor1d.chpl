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

    proc newIntersect1d(a1: [?aD] int, b1: [aD] int, assume_unique: string) {
      if assume_unique == "False" {
        var (a, _)  = uniqueSort(a1);
        var (b, _)  = uniqueSort(b1);
        var aux2 = concatset(a,b);
        var aux_sort_indices = radixSortLSD_ranks(aux2);
        var aux = aux2[aux_sort_indices];

        var mask = sliceEnd(aux) == sliceStart(aux);

        var temp = sliceEnd(aux);
        var int1d = boolIndexer(temp, mask);

        return int1d;
      } else {
        ref a = a1;
        ref b = b1;
        var aux2 = concatset(a,b);
        var aux_sort_indices = radixSortLSD_ranks(aux2);
        var aux = aux2[aux_sort_indices];

        var mask = sliceEnd(aux) == sliceStart(aux);

        var temp = sliceEnd(aux);
        var int1d = boolIndexer(temp, mask);

        return int1d;
      }
    }

    proc sliceStart(a: [?aD] ?t) {
      return sliceIndex(a, 1, a.size, 1);
    }

    proc sliceEnd(a: [?aD] ?t) {
      return sliceIndex(a, 0, a.size - 1, 1);
    }
      
    proc newSetxor1d(a1: [?aD] int, b1: [aD] int, assume_unique: string) {
      if assume_unique == "False" {
        var (a, _)  = uniqueSort(a1);
        var (b, _)  = uniqueSort(b1);

        var aux2 = concatset(a,b);
        var aux_sort_indices = radixSortLSD_ranks(aux2);
        var aux = aux2[aux_sort_indices];

        var sliceComp = sliceStart(aux) != sliceEnd(aux);//(sliceIndex(aux,1,aux.size,1) != sliceIndex(aux,0,aux.size-1,1));
        var flag = concatset([true],sliceComp);
        var flag2 = concatset(flag, [true]);

        var mask = sliceStart(flag2) & sliceEnd(flag2);//sliceIndex(flag2,1,flag2.size,1) & sliceIndex(flag2,0,flag2.size-1,1);

        var ret = boolIndexer(aux, mask);

        return ret;
      } else {
        ref a  = a1;
        ref b  = b1;

        var aux2 = concatset(a,b);
        var aux_sort_indices = radixSortLSD_ranks(aux2);
        var aux = aux2[aux_sort_indices];

        var sliceComp = sliceStart(aux) != sliceEnd(aux);//(sliceIndex(aux,1,aux.size,1) != sliceIndex(aux,0,aux.size-1,1));
        var flag = concatset([true],sliceComp);
        var flag2 = concatset(flag, [true]);

        var mask = sliceStart(flag2) & sliceEnd(flag2);//sliceIndex(flag2,1,flag2.size,1) & sliceIndex(flag2,0,flag2.size-1,1);

        var ret = boolIndexer(aux, mask);

        return ret;
      }
      
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

    proc boolIndexer(a: [?aD] ?t, truth: [aD] bool) {
        var iv: [truth.domain] int = (+ scan truth);
        var pop = iv[iv.size-1];
        var ret = makeDistArray(pop, int);

        forall (i, eai) in zip(a.domain, a) with (var agg = newDstAggregator(int)) {
          if (truth[i]) {
            agg.copy(ret[iv[i]-1], eai);
          }
        }
        return ret;
    }

    proc newSetdiff1d(a: [?aD] int, b: [?bD] int, assume_unique: string) {
      if assume_unique == "False" {
        var (a1, _)  = uniqueSort(a);
        var (b1, _)  = uniqueSort(b);
        var truth = makeDistArray(a1.size, bool);
        truth = in1dSort(a1,b1);
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
        truth = in1dSort(a,b);
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