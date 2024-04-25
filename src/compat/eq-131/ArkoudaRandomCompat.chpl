module ArkoudaRandomCompat {
  use Random.PCGRandom only PCGRandomStream;

  private proc is1DRectangularDomain(d) param do
    return d.isRectangular() && d.rank == 1;

  record randomStream {
    type eltType = int;
    forwarding var r: shared PCGRandomStream(eltType);
    proc init(type t) {
      eltType = t;
      r = new shared PCGRandomStream(eltType);
    }
    proc init(type t, seed) {
      eltType = t;
      r = new shared PCGRandomStream(eltType, seed);
    }
    proc ref fill(ref arr: []) where arr.isRectangular() {
      r.fillRandom(arr);
    }
    proc ref fill(ref arr: [], min: arr.eltType, max: arr.eltType) where arr.isRectangular() {
      r.fillRandom(arr, min, max);
    }
    proc ref permute(const ref arr: [?d] ?t): [] t  where is1DRectangularDomain(d) && isCoercible(this.eltType, d.idxType) {
      return r.permutation(arr);
    }
    proc ref permute(d: domain): [] d.idxType  where is1DRectangularDomain(d) && isCoercible(this.eltType, d.idxType) {
      // unfortunately there isn't a domain permutation function so we will create an array to permute
      var domArr: [d] d.idxType = d;
      r.permutation(domArr);
      return domArr;
    }
    proc ref sample(d: domain, n: int, withReplacement = false): [] d.idxType throws  where is1DRectangularDomain(d) {
      return choiceUniform(r, d, n, withReplacement);

      /* _choice branch for uniform distribution */
      proc choiceUniform(stream, X: domain, size: ?sizeType, replace: bool) throws
      {
        use Set;
        use Math;

        const low = X.low,
              stride = abs(X.stride);

        if isNothingType(sizeType) {
          // Return 1 sample
          var randVal = stream.getNext(resultType=int, 0, X.sizeAs(X.idxType)-1);
          var randIdx = X.dim(0).orderToIndex(randVal);
          return randIdx;
        } else {
          // Return numElements samples

          // Compute numElements for tuple case
          var m = 1;
          if isDomainType(sizeType) then m = size.size;

          var numElements = if isDomainType(sizeType) then m
                            else if isIntegralType(sizeType) then size:int
                            else compilerError('choice() size type must be integral or tuple of ranges');

          // Return N samples
          var samples: [0..<numElements] int;

          if replace {
            for sample in samples {
              var randVal = stream.getNext(resultType=int, 0, X.sizeAs(X.idxType)-1);
              var randIdx = X.dim(0).orderToIndex(randVal);
              sample = randIdx;
            }
          } else {
            if numElements < log2(X.sizeAs(X.idxType)) {
              var indices: set(int);
              var i: int = 0;
              while i < numElements {
                var randVal = stream.getNext(resultType=int, 0, X.sizeAs(X.idxType)-1);
                if !indices.contains(randVal) {
                  var randIdx = X.dim(0).orderToIndex(randVal);
                  samples[i] = randIdx;
                  indices.add(randVal);
                  i += 1;
                }
              }
            } else {
              var indices: [X] int = X;
              stream.shuffle(indices);
              for i in samples.domain {
                samples[i] = (indices[X.dim(0).orderToIndex(i)]);
              }
            }
          }
          if isIntegralType(sizeType) {
            return samples;
          } else if isDomainType(sizeType) {
            return reshape(samples, size);
          }
        }
      }
    }
    proc ref next(): eltType do return r.getNext();
    proc skipTo(n: int) do try! r.skipToNth(n);
  }

  proc sample(arr: [?d] ?t, n: int, withReplacement: bool): [] t throws {
    var r = new randomStream(int);
    return r.r.choice(arr, size=n, replace=withReplacement);
  }
}
