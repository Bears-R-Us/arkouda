module RandArray {
  use Random;
  use SegmentedArray;
  use ServerErrorStrings;
  use Map;
  use SipHash;

  proc fillInt(a:[] ?t, const aMin, const aMax) where isIntType(t) {
    coforall loc in Locales {
      on loc {
        ref myA = a.localSlice[a.localSubdomain()];
        fillRandom(myA);
        [ai in myA] if (ai < 0) { ai = -ai; }
        if (aMax > aMin) {
          const modulus = aMax - aMin;
          [x in myA] x = ((x % modulus) + aMin):t;
          //myA = (myA % modulus) + aMin:t;
        }
      }
    }
  }

  proc fillUInt(a:[] ?t, const aMin, const aMax) where isUintType(t) {
    coforall loc in Locales {
      on loc {
        ref myA = a.localSlice[a.localSubdomain()];
        fillRandom(myA);
        if (aMax > aMin) {
          const modulus = aMax - aMin;
          [x in myA] x = ((x % modulus) + aMin):t;
          //myA = (myA % modulus) + aMin:t;
        }
      }
    }
  }

  proc fillReal(a:[] real, const aMin=0, const aMax=1) {
    coforall loc in Locales {
      on loc {
        ref myA = a.localSlice[a.localSubdomain()];
        fillRandom(myA);
        const scale = aMax - aMin;
        myA = scale*myA + aMin;
      }
    }
  }

  proc fillBool(a:[] bool) {
    coforall loc in Locales {
      on loc {
        ref myA = a.localSlice[a.localSubdomain()];
        fillRandom(myA);
      }
    }
  }

  proc fillNormal(a:[?D] real) {
    var u1:[D] real;
    var u2:[D] real;
    fillRandom(u1);
    fillRandom(u2);
    a = sqrt(-2*log(u1))*cos(2*pi*u2);
  }

  enum charSet {
    Uppercase,
    Lowercase,
    Numeric,
    Printable,
    Binary
  }

  var charBounds: map(keyType=charSet, valType=2*int, parSafe=false);
  charBounds[charSet.Uppercase] = (65, 91);
  charBounds[charSet.Lowercase] = (97, 123);
  charBounds[charSet.Numeric] = (48, 58);
  charBounds[charSet.Printable] = (32, 127);
  charBounds[charSet.Binary] = (0, 0);

  proc newRandStringsUniformLength(const n: int, const minLen: int, const maxLen: int, characters:charSet = charSet.Uppercase) throws {
    if (n < 0) || (minLen < 0) || (maxLen < minLen) {
      throw new owned ArgumentError();
    }
    var lengths = makeDistArray(n, int);
    fillInt(lengths, minLen+1, maxLen+1);
    const nBytes = + reduce lengths;
    var segs = (+ scan lengths) - lengths;
    var vals = makeDistArray(nBytes, uint(8));
    var (lb, ub) = charBounds[characters];
    fillUInt(vals, lb, ub);
    // Strings are null-terminated
    [(s, l) in zip(segs, lengths)] vals[s+l-1] = 0:uint(8);
    return (segs, vals);
  }

  proc newRandStringsLogNormalLength(const n: int, const logMean: numeric, const logStd: numeric, characters:charSet = charSet.Uppercase) throws {
    if (n < 0) || (logStd <= 0) {
      throw new owned ArgumentError();
    }
    var ltemp = makeDistArray(n, real);
    fillNormal(ltemp);
    ltemp = exp(logMean + logStd*ltemp);
    var lengths:[ltemp.domain] int = [l in ltemp] ceil(l):int;
    const nBytes = + reduce lengths;
    var segs = (+ scan lengths) - lengths;
    var vals = makeDistArray(nBytes, uint(8));
    var (lb, ub) = charBounds[characters];
    fillUInt(vals, lb, ub);
    // Strings are null-terminated
    [(s, l) in zip(segs, lengths)] vals[s+l-1] = 0:uint(8);
    return (segs, vals);
  }
}
