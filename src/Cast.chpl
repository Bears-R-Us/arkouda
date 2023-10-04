module Cast {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Reflection;
  use SegmentedString;
  use ServerErrors;
  use Logging;
  use CommAggregation;
  use ServerConfig;

  use ArkoudaBigIntCompat;
  use ArkoudaMathCompat;
  
  private config const logLevel = ServerConfig.logLevel;
  const castLogger = new Logger(logLevel);
  
  proc castGenSymEntry(gse: borrowed GenSymEntry, st: borrowed SymTab, type fromType, 
                                             type toType): string throws {
    const before = toSymEntry(gse, fromType);
    const name = st.nextName();
    var after = st.addEntry(name, before.size, toType);
    try {
      after.a = before.a : toType;
    } catch e: IllegalArgumentError {
      var errorMsg = "bad value in cast from %s to %s".doFormat(fromType:string, 
                                                       toType:string);
      castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);   
      return "Error: %s".doFormat(errorMsg);
    }

    var returnMsg = "created " + st.attrib(name);
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),returnMsg);
    return returnMsg;
  }

  proc castGenSymEntryToBigInt(gse: borrowed GenSymEntry, st: borrowed SymTab, type fromType): string throws {
    const before = toSymEntry(gse, fromType);
    const name = st.nextName();
    var tmp = makeDistArray(before.size, bigint);
    try {
      if fromType == bigint {
        tmp = before.a;
      }
      else {
        tmp = before.a:bigint;
      }
    } catch e: IllegalArgumentError {
      var errorMsg = "bad value in cast from %s to bigint".doFormat(fromType:string);
      castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
      return "Error: %s".doFormat(errorMsg);
    }
    var after = st.addEntry(name, createSymEntry(tmp));

    var returnMsg = "created " + st.attrib(name);
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),returnMsg);
    return returnMsg;
  }

  proc castGenSymEntryToString(gse: borrowed GenSymEntry, st: borrowed SymTab, 
                                                       type fromType): string throws {
    const before = toSymEntry(gse, fromType);
    const oname = st.nextName();
    var segments = st.addEntry(oname, before.size, int);
    var strings = makeDistArray(before.a.domain, string);
    if fromType == real {
      try {
          forall (s, v) in zip(strings, before.a) {
              s = "%.17r".doFormat(v);
          }
      } catch e {
          var errorMsg = "could not convert float64 value to decimal representation";
          castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);   
          return "Error: %s".doFormat(errorMsg);
      }
    } else {
      try {
          strings = [s in before.a] s : string;
      } catch e: IllegalArgumentError {
          var errorMsg = "bad value in cast from %s to string".doFormat(fromType:string);
          castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);   
          return "Error: %s".doFormat(errorMsg);
      }
    }
    const byteLengths = [s in strings] s.numBytes + 1;
    // check there's enough room to create a copy for scan and throw if creating a copy would go over memory limit
    overMemLimit(numBytes(uint(8)) * byteLengths.size);
    segments.a = (+ scan byteLengths) - byteLengths;
    const totBytes = + reduce byteLengths;
    const vname = st.nextName();
    var values = st.addEntry(vname, totBytes, uint(8));
    ref va = values.a;
    forall (o, s) in zip(segments.a, strings) with (var agg = newDstAggregator(uint(8))) {
      for (b, i) in zip(s.bytes(), 0..) {
        agg.copy(va[o+i], b);
      }
    }

    var returnMsg ="created " + st.attrib(oname) + "+created " + st.attrib(vname);
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),returnMsg);
    return returnMsg;
  }

  enum ErrorMode {
    strict,
    ignore,
    return_validity,
  }

  inline proc stringToNumericStrict(ref values, rng, type toType): toType throws {
    if toType == bool {
      return interpretAsString(values, rng).toLower() : toType;
    } else {
      return interpretAsString(values, rng) : toType;
    }
  }

  inline proc stringToNumericIgnore(ref values, rng, type toType): toType {
    var num: toType;
    try {
      num = stringToNumericStrict(values, rng, toType);
    } catch {
      if toType == real {
        num = nan;
      } else if toType == int {
        // Use pandas.NaT, i.e. -2**63, as NaN for int
        num = min(int);
      }
      // Other types remain zero on error
    }
    return num;
  }

  inline proc stringToNumericReturnValidity(ref values, rng, type toType): (toType, bool) {
    var num: toType;
    var valid = true;
    try {
      num = stringToNumericStrict(values, rng, toType);
    } catch {
      if toType == real {
        num = nan;
      } else if toType == int {
        // Use pandas.NaT, i.e. -2**63, as NaN for int
        num = min(int);
      }
      // Other types remain zero on error
      valid = false;
    }
    return (num, valid);
  }

  
  proc castStringToSymEntry(s: SegString, st: borrowed SymTab, type toType, errors: ErrorMode): string throws {
      use SegmentedComputation;
      ref oa = s.offsets.a;
      ref va = s.values.a;
      const name = st.nextName();
      var entry = st.addEntry(name, s.size, toType);
      var returnMsg = "created " + st.attrib(name);
      select errors {
        when ErrorMode.strict {
          entry.a = computeOnSegments(oa, va, SegFunction.StringToNumericStrict, toType);
        }
        when ErrorMode.ignore {
          entry.a = computeOnSegments(oa, va, SegFunction.StringToNumericIgnore, toType);
        }
        when ErrorMode.return_validity {
          var valWithFlag: [entry.a.domain] (toType, bool) = computeOnSegments(oa, va, SegFunction.StringToNumericReturnValidity, (toType, bool));
          const vname = st.nextName();
          var valid = st.addEntry(vname, s.size, bool);
          forall (n, v, vf) in zip(entry.a, valid.a, valWithFlag) {
            (n, v) = vf;
          }
          returnMsg += "+created " + st.attrib(vname);
        }
      }
      castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),returnMsg);
      return returnMsg;
  }

    proc castStringToBigInt(s: SegString, st: borrowed SymTab, errors: ErrorMode): string throws {
      use SegmentedComputation;
      ref oa = s.offsets.a;
      ref va = s.values.a;
      var returnMsg = "";
      const name = st.nextName();
      // do something like segmented computation w/o the aggregation
      select errors {
        when ErrorMode.strict {
          var entry = st.addEntry(name, createSymEntry(computeOnSegmentsWithoutAggregation(oa, va, SegFunction.StringToNumericStrict, bigint)));
          returnMsg = "created " + st.attrib(name);
        }
        when ErrorMode.ignore {
          var entry = st.addEntry(name, createSymEntry(computeOnSegmentsWithoutAggregation(oa, va, SegFunction.StringToNumericIgnore, bigint)));
          returnMsg = "created " + st.attrib(name);
        }
        when ErrorMode.return_validity {
          var valWithFlag = computeOnSegmentsWithoutAggregation(oa, va, SegFunction.StringToNumericReturnValidity, (bigint, bool));
          const vname = st.nextName();
          var valid = st.addEntry(vname, s.size, bool);
          var tmp = makeDistArray(s.size, bigint);
          forall (t, v, vf) in zip(tmp, valid.a, valWithFlag) {
            (t, v) = vf;
          }
          var entry = st.addEntry(name, createSymEntry(tmp));
          returnMsg = "created " + st.attrib(name);
          returnMsg += "+created " + st.attrib(vname);
        }
      }
      castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),returnMsg);
      return returnMsg;
  }
}
