module RandArray {
  use Reflection;
  use ServerErrors;
  use Logging;
  use Random;
  use ArkoudaRandomCompat;
  use ServerErrorStrings;
  use MultiTypeSymEntry;
  use Map;
  use SipHash;
  use ServerConfig;
  private use IO;
  
  private config const logLevel = ServerConfig.logLevel;
  const raLogger = new Logger(logLevel);

  proc fillInt(a:[] ?t, const aMin: t, const aMax: t, const seedStr:string="None") throws where isIntType(t) {
      if (seedStr.toLower() == "none") {
        //Subtracting 1 from aMax to make the value exclusive to follow numpy standard.
        fillRandom(a, aMin, aMax-1);
      } else {
        var seed = (seedStr:int) + here.id;
        //Subtracting 1 from aMax to make the value exclusive to follow numpy standard.
        fillRandom(a, aMin, aMax-1, seed);
      }
  }

  proc fillUInt(a:[] ?t, const aMin: t, const aMax: t, const seedStr:string="None") throws where isUintType(t) {
      if (seedStr.toLower() == "none") {
        //Subtracting 1 from aMax to make the value exclusive to follow numpy standard.
        fillRandom(a, aMin, aMax-1);
      } else {
        var seed = (seedStr:int) + here.id;
        //Subtracting 1 from aMax to make the value exclusive to follow numpy standard.
        fillRandom(a, aMin, aMax-1, seed);
      }
  }

  proc fillReal(a:[] real, const aMin:numeric=0.0, const aMax:numeric=1.0, const seedStr:string="None") throws {
    if (seedStr.toLower() == "none") {
      fillRandom(a);
    } else {
      var seed = (seedStr:int) + here.id;
      fillRandom(a, seed);
    }
    const scale = aMax - aMin;
    a = scale*a + aMin;
  }

  proc fillBool(a:[] bool, const seedStr:string="None") throws {
    if (seedStr.toLower() == "none") {
      fillRandom(a);
    } else {
      var seed = (seedStr:int) + here.id;
      fillRandom(a, seed);
    }
  }

  proc fillNormal(a:[?D] real, const seedStr:string="None") throws {
    var u1:[D] real;
    var u2:[D] real;
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

  var charBounds: map(keyType=charSet, valType=2*int, parSafe=false);
  charBounds[charSet.Uppercase] = (65, 91);
  charBounds[charSet.Lowercase] = (97, 123);
  charBounds[charSet.Numeric] = (48, 58);
  charBounds[charSet.Printable] = (32, 127);
  charBounds[charSet.Binary] = (0, 0);

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
    var lengths:[ltemp.domain] int = [l in ltemp] ceil(l):int;
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
