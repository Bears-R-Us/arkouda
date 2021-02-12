module CastMsg {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Reflection;
  use SegmentedArray;
  use Errors;
  use Logging;
  use SysError;
  use ServerErrorStrings;
  use ServerConfig;
  use CommAggregation;

  const castLogger = new Logger();
  if v {
      castLogger.level = LogLevel.DEBUG;
  } else {
      castLogger.level = LogLevel.INFO;    
  }

  proc castMsg(cmd: string, payload: string, st: borrowed SymTab): string throws {
    use ServerConfig; // for string.splitMsgToTuple
    param pn = Reflection.getRoutineName();
    var (name, objtype, targetDtype, opt) = payload.splitMsgToTuple(4);
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
          "name: %s obgtype: %t targetDtype: %t opt: %t".format(
                                                 name,objtype,targetDtype,opt));
    select objtype {
      when "pdarray" {
        var gse: borrowed GenSymEntry = st.lookup(name);
        select (gse.dtype, targetDtype) {
        when (DType.Int64, "int64") {
          return castGenSymEntry(gse, st, int, int);
        }
        when (DType.Int64, "uint8") {
          return castGenSymEntry(gse, st, int, uint(8));
        }
        when (DType.Int64, "float64") {
          return castGenSymEntry(gse, st, int, real);
        }
        when (DType.Int64, "bool") {
          return castGenSymEntry(gse, st, int, bool);
        }
        when (DType.Int64, "str") {
          return castGenSymEntryToString(gse, st, int);
        }
        when (DType.UInt8, "int64") {
          return castGenSymEntry(gse, st, uint(8), int);        
        }
        when (DType.UInt8, "uint8") {
          return castGenSymEntry(gse, st, uint(8), uint(8));        
        }
        when (DType.UInt8, "float64") {
          return castGenSymEntry(gse, st, uint(8), real);          
        }
        when (DType.UInt8, "bool") {
          return castGenSymEntry(gse, st, uint(8), bool);                  
        }
        when (DType.UInt8, "str") {
          return castGenSymEntryToString(gse, st, uint(8));
        }
        when (DType.Float64, "int64") {
          return castGenSymEntry(gse, st, real, int);                  
        }
        when (DType.Float64, "uint8") {
          return castGenSymEntry(gse, st, real, uint(8));
        }
        when (DType.Float64, "float64") {
          return castGenSymEntry(gse, st, real, real);
        }
        when (DType.Float64, "bool") {
          return castGenSymEntry(gse, st, real, bool);
        }
        when (DType.Float64, "str") {
          return castGenSymEntryToString(gse, st, real);
        }
        when (DType.Bool, "int64") {
          return castGenSymEntry(gse, st, bool, int);
        }
        when (DType.Bool, "uint8") {
          return castGenSymEntry(gse, st, bool, uint(8));
        }
        when (DType.Bool, "float64") {
          return castGenSymEntry(gse, st, bool, real);
        }
        when (DType.Bool, "bool") {
          return castGenSymEntry(gse, st, bool, bool);
        }
        when (DType.Bool, "str") {
          return castGenSymEntryToString(gse, st, bool);
        }
        otherwise {
            var errorMsg = notImplementedError(pn,gse.dtype:string,":",targetDtype);
            castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                    
            return errorMsg;
        }
        }
      }
      when "str" {
        const (segName, valName) = name.splitMsgToTuple("+", 2);
        const strings = new owned SegString(segName, valName, st);
        select targetDtype {
          when "int64" {
            return castStringToSymEntry(strings, st, int);
          }
          when "uint8" {
            return castStringToSymEntry(strings, st, uint(8));
          }
          when "float64" {
            return castStringToSymEntry(strings, st, real);
          }
          when "bool" {
            return castStringToSymEntry(strings, st, bool);
          }
          otherwise {
            var errorMsg = notImplementedError(pn,"str",":",targetDtype);
            castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return errorMsg;
          }
          }
      }
      otherwise {
        var errorMsg = notImplementedError(pn,objtype);
        castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                      
        return errorMsg;
      }
      }
  }

  proc castGenSymEntry(gse: borrowed GenSymEntry, st: borrowed SymTab, type fromType, 
                                             type toType): string throws {
    const before = toSymEntry(gse, fromType);
    const name = st.nextName();
    var after = st.addEntry(name, before.size, toType);
    try {
      after.a = before.a : toType;
    } catch e: IllegalArgumentError {
      var errorMsg = "bad value in cast from %s to %s".format(fromType:string, 
                                                       toType:string);
      castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);   
      return "Error: %s".format(errorMsg);
    }

    var returnMsg = "created " + st.attrib(name);
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),returnMsg);
    return returnMsg;
  }

  proc castGenSymEntryToString(gse: borrowed GenSymEntry, st: borrowed SymTab, 
                                                       type fromType): string throws {
    const before = toSymEntry(gse, fromType);
    const oname = st.nextName();
    var segments = st.addEntry(oname, before.size, int);
    var strings: [before.aD] string;
    if fromType == real {
      try {
          forall (s, v) in zip(strings, before.a) {
              s = "%.17r".format(v);
          }
      } catch e {
          var errorMsg = "could not convert float64 value to decimal representation";
          castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);   
          return "Error: %s".format(errorMsg);
      }
    } else {
      try {
          strings = [s in before.a] s : string;
      } catch e: IllegalArgumentError {
          var errorMsg = "bad value in cast from %s to string".format(fromType:string);
          castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);   
          return "Error: %s".format(errorMsg);
      }
    }
    const byteLengths = [s in strings] s.numBytes + 1;
    segments.a = (+ scan byteLengths) - byteLengths;
    const totBytes = + reduce byteLengths;
    const vname = st.nextName();
    var values = st.addEntry(vname, totBytes, uint(8));
    ref va = values.a;
    forall (o, s) in zip(segments.a, strings) with (var agg = newDstAggregator(uint(8))) {
      for (i, b) in zip(0.., s.bytes()) {
        agg.copy(va[o+i], b);
      }
    }

    var returnMsg ="created " + st.attrib(oname) + "+created " + st.attrib(vname);
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),returnMsg);
    return returnMsg;
  }

  proc castStringToSymEntry(s: SegString, st: borrowed SymTab, type toType): string throws {
      ref oa = s.offsets.a;
      ref va = s.values.a;
      const name = st.nextName();
      var entry = st.addEntry(name, s.size, toType);
    
      const highInd = s.offsets.aD.high;
      try {
          forall (i, o, e) in zip(s.offsets.aD, s.offsets.a, entry.a) {
              const start = o;
              var end: int;

              if (i == highInd) {
              end = s.nBytes - 1;
              } else {
                   end = oa[i+1] - 1;
              }
              e = interpretAsString(va[start..end]) : toType;
          }
      } catch e: IllegalArgumentError {
          var errorMsg = "bad value in cast from string to %s".format(toType:string);
          castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);   
          return "Error: %s".format(errorMsg);
      }

      var returnMsg = "created " + st.attrib(name);
      castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),returnMsg);
      return returnMsg;
  }
  
}