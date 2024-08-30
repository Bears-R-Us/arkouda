module RandArray {
  use Reflection;
  use ServerErrors;
  use Logging;
  use Random;
  use ServerErrorStrings;
  use MultiTypeSymEntry;
  use CommAggregation;
  use SipHash;
  use ServerConfig;
  private use IO;
  use Math;
  use Map;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const raLogger = new Logger(logLevel, logChannel);

  proc fillInt(ref a:[] ?t, const aMin: t, const aMax: t, const seedStr:string="None") throws where isIntType(t) {
      if (seedStr.toLower() == "none") {
        //Subtracting 1 from aMax to make the value exclusive to follow numpy standard.
        fillRandom(a, aMin, aMax-1);
      } else {
        var seed = (seedStr:int) + here.id;
        //Subtracting 1 from aMax to make the value exclusive to follow numpy standard.
        fillRandom(a, aMin, aMax-1, seed);
      }
  }

  proc fillUInt(ref a:[] ?t, const aMin: t, const aMax: t, const seedStr:string="None") throws where isUintType(t) {
      if (seedStr.toLower() == "none") {
        //Subtracting 1 from aMax to make the value exclusive to follow numpy standard.
        fillRandom(a, aMin, aMax-1);
      } else {
        var seed = (seedStr:int) + here.id;
        //Subtracting 1 from aMax to make the value exclusive to follow numpy standard.
        fillRandom(a, aMin, aMax-1, seed);
      }
  }


  proc fillNormal(ref a:[?D] real, const seedStr:string="None") throws {
    // uses Box–Muller transform
    // generates values drawn from the standard normal distribution using
    // 2 uniformly distributed random numbers
    var u1 = makeDistArray(D, real);
    var u2 = makeDistArray(D, real);
    if (seedStr.toLower() == "none") {
      fillRandom(u1);
      fillRandom(u2);
    } else {
      var seed = (seedStr:int);
      fillRandom(u1, seed);
      fillRandom(u2, seed+1);
    }
    a = sqrt(-2*log(u1))*cos(2*pi*u2);
  }

  enum charSet {
    Uppercase,
    Lowercase,
    Numeric,
    Printable,
    Binary
  }

  proc str2CharSet(str: string): charSet {
    var ret: charSet;
    select str.toLower() {
      when "uppercase" {
        ret = charSet.Uppercase;
      }
      when "lowercase" {
        ret = charSet.Lowercase;
      }
      when "numeric" {
        ret = charSet.Numeric;
      }
      when "printable" {
        ret = charSet.Printable;
      }
      when "binary" {
        ret = charSet.Binary;
      }
      otherwise {
        ret = charSet.Uppercase;
      }
    }
    return ret;
  }

  /*
    Return an array of `n` random samples from the input array `a` with a
    corresponding array of weights for each element in `a`.

    The weights array must be the same length as `a` and have at least one
    non-zero element.

    If `withReplacement` is `false`, `n` must be less than or equal to the
    length of `a`. In this mode, the weights array still designates how likely
    a given element is to be sampled, but after being sampled, its weight is
    treated as 0 for the remainder of the sampling process.

    When `withReplacement` is `true`, `n` can be any positive integer and the
    weights array remains constant throughout the sampling process.

    :arg rs: The random stream to use (must be a real-valued stream)
    :arg a: The input array to sample from
    :arg weights: The weights corresponding to each element in `a`
    :arg n: The number of samples to draw
    :arg withReplacement: Whether to sample with replacement or not

    :returns: An array of `n` samples from `a`
  */
  proc randSampleWeights(
    ref rs: randomStream(real),
    const ref a: [?d] ?t,
    const ref weights: [d] real,
    n: int,
    withReplacement: bool
  ): [] t throws {
    // get n random indices from a/weights's domain
    // TODO: index into 'a' directly rather than creating a new (potentially large) index array
    const indices = sampleDomWeighted(rs, n, weights, withReplacement);

    // create a new array to hold the samples
    var ret = makeDistArray({0..<n}, t);

    // copy the samples into the new array
    forall idx in ret.domain with (var agg = newSrcAggregator(t)) do
      agg.copy(ret[idx], a[indices[idx]]);

    return ret;
  }

  // helper method for 'randSampleWeights'
  // this is a slight simplification of 'randomStream.sample' from the Random module
  proc sampleDomWeighted(
    ref rs: randomStream(real), n: int, const ref weights: [?dw] real, withReplacement: bool
  ): [] int throws {
    import Sort;
    import Search;

    if dw.size < 1 then
      throw new IllegalArgumentError("Cannot sample from an empty domain");

    if n < 1 || (n > dw.size && !withReplacement) then
      throw new IllegalArgumentError("Number of samples must be >= 1 and <= weights.size when withReplacement=false");

    // compute the normalized cumulative weights
    var cw = + scan weights;
    cw /= cw[dw.last];

    if !Sort.isSorted(cw) then
      throw new IllegalArgumentError("'weights' cannot contain negative values");

    if cw[dw.last] <= 1e-15 then
      throw new IllegalArgumentError("'weights' must contain at least one non-zero value");

    const dOut = {0..<n};
    var samples: [dOut] int;

    if withReplacement {
      // use binary search to sample `n` indices
      for i in dOut {
        const (_, ii) = Search.binarySearch(cw, rs.next() /* sample between 0.0 and 1.0 */);
        samples[i] = ii;
      }
    } else {
      var weightsCopy = weights,
          indices: domain(int, parSafe=false),
          i = 0,
          ii = 0;

      while i < n {
        // sample an index that hasn't been sampled yet
        do {
          (_, ii) = Search.binarySearch(cw, rs.next() /* sample between 0.0 and 1.0 */);
        } while indices.contains(ii);

        // add the sampled index to the list of indices and the list of samples
        weightsCopy[ii] = 0;
        indices += ii;
        samples[i] = ii;
        i += 1;

        // recompute the normalized cumulative weights
        cw = + scan weightsCopy;
        cw /= cw[dw.last];
      }
    }

    return samples;
  }

  var charBounds: map(keyType=charSet, valType=2*int);
  try! {
    charBounds[charSet.Uppercase] = (65, 91);
    charBounds[charSet.Lowercase] = (97, 123);
    charBounds[charSet.Numeric] = (48, 58);
    charBounds[charSet.Printable] = (32, 127);
    charBounds[charSet.Binary] = (0, 0);
  }

  proc newRandStringsUniformLength(const n: int,
                                   const minLen: int, 
                                   const maxLen: int,
                                   characters:charSet = charSet.Uppercase,
                                   const seedStr:string="None") throws {
    if (n < 0) || (minLen < 0) || (maxLen < minLen) {  
        raLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                  "Incompatible arguments: n and minLen must be > 0 and maxLen < minLen"); 
        throw new owned ArgumentError();                     
    }
    var lengths = makeDistArray(n, int);
    fillInt(lengths, minLen+1, maxLen+1, seedStr=seedStr);
    const nBytes = + reduce lengths;
    // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
    overMemLimit(numBytes(int) * lengths.size);
    var segs = (+ scan lengths) - lengths;
    var vals = makeDistArray(nBytes, uint(8));
    var (lb, ub) = charBounds[characters];
    fillUInt(vals, lb:uint(8), ub:uint(8), seedStr=seedStr);
    // Strings are null-terminated
    [(s, l) in zip(segs, lengths)] vals[s+l-1] = 0:uint(8);
    return (segs, vals);
  }

  proc newRandStringsLogNormalLength(const n: int,
                                     const logMean: numeric, 
                                     const logStd: numeric,
                                     characters:charSet = charSet.Uppercase,
                                     const seedStr:string="None") throws {
    if (n < 0) || (logStd <= 0) {
        raLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                     "Incompatible arguments: n must be > 0 and logStd <= 0");      
        throw new owned ArgumentError();
    }
    var ltemp = makeDistArray(n, real);
    fillNormal(ltemp, seedStr=seedStr);
    ltemp = exp(logMean + logStd*ltemp);
    var lengths = makeDistArray(ltemp.domain, int);
    lengths = [l in ltemp] ceil(l):int;
    const nBytes = + reduce lengths;
    // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
    overMemLimit(numBytes(int) * lengths.size);
    var segs = (+ scan lengths) - lengths;
    var vals = makeDistArray(nBytes, uint(8));
    var (lb, ub) = charBounds[characters];
    fillUInt(vals, lb:uint(8), ub:uint(8), seedStr=seedStr);
    // Strings are null-terminated
    [(s, l) in zip(segs, lengths)] vals[s+l-1] = 0:uint(8);
    return (segs, vals);
  }
}
