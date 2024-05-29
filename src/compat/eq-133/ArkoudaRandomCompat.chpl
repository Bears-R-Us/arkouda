module ArkoudaRandomCompat {
  public use Random;

  private proc is1DRectangularDomain(d) param do
    return d.isRectangular() && d.rank == 1;

  proc sample(arr: [?d] ?t, n: int, withReplacement: bool): [] t throws {
    var r = new randomStream(int);
    return r.choice(arr, size=n, replace=withReplacement);
  }
  proc ref randomStream.skipTo(n: int) do try! this.skipToNth(n);
  proc ref randomStream.permute(const ref arr: [?d] ?t): [] t  where is1DRectangularDomain(d) && isCoercible(this.eltType, d.idxType) {
    return this.permutation(arr);
  }
  proc ref randomStream.permute(d: domain(?)): [] d.idxType  where is1DRectangularDomain(d) && isCoercible(this.eltType, d.idxType) {
    // unfortunately there isn't a domain permutation function so we will create an array to permute
    var domArr: [d] d.idxType = d;
    this.permutation(domArr);
    return domArr;
  }
  proc ref randomStream.sample(d: domain(?), n: int, withReplacement = false): [] d.idxType throws where is1DRectangularDomain(d) && isCoercible(this.eltType, d.idxType) {
    return choiceUniform(this, d, n, withReplacement);
  }
  proc ref randomStream.sample(const x: [?dom], size:?sizeType=none, replace=true) throws {
      var idx = choiceUniform(this, dom, size, replace);
      return x[idx];
    }
  proc ref randomStream.next() do return this.getNext();
  proc ref randomStream.next(min: eltType, max: eltType): eltType do return r.getNext(min, max);

  proc choiceUniform(ref stream, X: domain(?), size: ?sizeType, replace: bool) throws
  {
    use Math;
    const low = X.low,
          stride = abs(X.stride);

    if isNothingType(sizeType) {
      // Return 1 sample
      var randVal;
      // TODO: removed first branch of this conditional after PCG/NPBRandomStream deprecations
      if __primitive("method call and fn resolves", stream, "getNext", X.idxType) {
        randVal = stream.getNext(resultType=X.idxType, 0, X.sizeAs(X.idxType)-1);
      } else {
        randVal = stream.getNext(0, X.sizeAs(X.idxType)-1);
      }
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
          var randVal;
          // TODO: removed first branch of this conditional after PCG/NPBRandomStream deprecations
          if __primitive("method call and fn resolves", stream, "getNext", X.idxType) {
            randVal = stream.getNext(resultType=X.idxType, 0, X.sizeAs(X.idxType)-1);
          } else {
            randVal = stream.getNext(0, X.sizeAs(X.idxType)-1);
          }
          var randIdx = X.dim(0).orderToIndex(randVal);
          sample = randIdx;
        }
      } else {
        if numElements < log2(X.sizeAs(X.idxType)) {
          var indices: domain(int, parSafe=false);
          var i: int = 0;
          while i < numElements {
            var randVal;
            // TODO: removed first branch of this conditional after PCG/NPBRandomStream deprecations
            if __primitive("method call and fn resolves", stream, "getNext", X.idxType) {
              randVal = stream.getNext(resultType=X.idxType, 0, X.sizeAs(X.idxType)-1);
            } else {
              randVal = stream.getNext(0, X.sizeAs(X.idxType)-1);
            }
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
