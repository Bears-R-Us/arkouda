module CastMsg {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Reflection;
  use SegmentedString;
  use ServerErrors;
  use Logging;
  use Message;
  use SysError;
  use ServerErrorStrings;
  use ServerConfig;
  use Cast;

  private config const logLevel = ServerConfig.logLevel;
  const castLogger = new Logger(logLevel);

  proc castMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    use ServerConfig; // for string.splitMsgToTuple
    param pn = Reflection.getRoutineName();
    var (name, objtype, targetDtype, opt) = payload.splitMsgToTuple(4);
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
          "name: %s obgtype: %t targetDtype: %t opt: %t".format(
                                                 name,objtype,targetDtype,opt));
    select objtype {
      when "pdarray" {
        var gse: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        select (gse.dtype, targetDtype) {
            when (DType.Int64, "int64") {
                return new MsgTuple(castGenSymEntry(gse, st, int, int), MsgType.NORMAL);
            }
            when (DType.Int64, "uint8") {
                return new MsgTuple(castGenSymEntry(gse, st, int, uint(8)), MsgType.NORMAL);
            }
            when (DType.Int64, "float64") {
                return new MsgTuple(castGenSymEntry(gse, st, int, real), MsgType.NORMAL);
            }
            when (DType.Int64, "bool") {
                return new MsgTuple(castGenSymEntry(gse, st, int, bool), MsgType.NORMAL);
            }
            when (DType.Int64, "str") {
                return new MsgTuple(castGenSymEntryToString(gse, st, int), MsgType.NORMAL);
            }
            when (DType.Int64, "uint64") {
              return new MsgTuple(castGenSymEntry(gse, st, int, uint), MsgType.NORMAL);
            }
            when (DType.UInt8, "int64") {
                return new MsgTuple(castGenSymEntry(gse, st, uint(8), int), MsgType.NORMAL);        
            }
            when (DType.UInt8, "uint8") {
                return new MsgTuple(castGenSymEntry(gse, st, uint(8), uint(8)), MsgType.NORMAL);        
            }
            when (DType.UInt8, "float64") {
                return new MsgTuple(castGenSymEntry(gse, st, uint(8), real), MsgType.NORMAL);          
            }
            when (DType.UInt8, "bool") {
                return new MsgTuple(castGenSymEntry(gse, st, uint(8), bool), MsgType.NORMAL);                 
            }
            when (DType.UInt8, "uint64") {
                return new MsgTuple(castGenSymEntry(gse, st, uint(8), uint), MsgType.NORMAL);                 
            }
            when (DType.UInt8, "str") {
                return new MsgTuple(castGenSymEntryToString(gse, st, uint(8)), MsgType.NORMAL);
            }
            when (DType.UInt64, "int64") {
              return new MsgTuple(castGenSymEntry(gse, st, uint, int), MsgType.NORMAL);
            }
            when (DType.UInt64, "uint8") {
              return new MsgTuple(castGenSymEntry(gse, st, uint, uint(8)), MsgType.NORMAL);
            }
            when (DType.UInt64, "uint64") {
                return new MsgTuple(castGenSymEntry(gse, st, uint, uint), MsgType.NORMAL);
            }
            when (DType.UInt64, "float") {
                return new MsgTuple(castGenSymEntry(gse, st, uint, real), MsgType.NORMAL);
            }
            when (DType.UInt64, "bool") {
                return new MsgTuple(castGenSymEntry(gse, st, uint, bool), MsgType.NORMAL);
            }
            when (DType.UInt64, "string") {
                return new MsgTuple(castGenSymEntryToString(gse, st, uint), MsgType.NORMAL);
            }
            when (DType.Float64, "int64") {
                return new MsgTuple(castGenSymEntry(gse, st, real, int), MsgType.NORMAL);                  
            }
            when (DType.Float64, "uint8") {
                return new MsgTuple(castGenSymEntry(gse, st, real, uint(8)), MsgType.NORMAL);
            }
            when (DType.Float64, "float64") {
                return new MsgTuple(castGenSymEntry(gse, st, real, real), MsgType.NORMAL);
            }
            when (DType.Float64, "bool") {
                return new MsgTuple(castGenSymEntry(gse, st, real, bool), MsgType.NORMAL);
            }
            when (DType.Float64, "uint64") {
              return new MsgTuple(castGenSymEntry(gse, st, real, uint), MsgType.NORMAL);
            }
            when (DType.Float64, "str") {
                return new MsgTuple(castGenSymEntryToString(gse, st, real), MsgType.NORMAL);
            }
            when (DType.Bool, "int64") {
                return new MsgTuple(castGenSymEntry(gse, st, bool, int), MsgType.NORMAL);
            }
            when (DType.Bool, "uint8") {
                return new MsgTuple(castGenSymEntry(gse, st, bool, uint(8)), MsgType.NORMAL);
            }
            when (DType.Bool, "uint64") {
                return new MsgTuple(castGenSymEntry(gse, st, bool, uint), MsgType.NORMAL);
            }
            when (DType.Bool, "float64") {
                return new MsgTuple(castGenSymEntry(gse, st, bool, real), MsgType.NORMAL);
            } 
            when (DType.Bool, "bool") {
                return new MsgTuple(castGenSymEntry(gse, st, bool, bool), MsgType.NORMAL);
            }
            when (DType.Bool, "str") {
                return new MsgTuple(castGenSymEntryToString(gse, st, bool), MsgType.NORMAL);
            }
            otherwise {
                var errorMsg = notImplementedError(pn,gse.dtype:string,":",targetDtype);
                castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                    
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
      }
      when "str" {
          const strings = getSegString(name, st);
          const errors = opt.toLower() : ErrorMode;
          select targetDtype {
              when "int64" {
                  return new MsgTuple(castStringToSymEntry(strings, st, int, errors), MsgType.NORMAL);
              }
              when "uint8" {
                  return new MsgTuple(castStringToSymEntry(strings, st, uint(8), errors), MsgType.NORMAL);
              }
              when "uint64" {
                  return new MsgTuple(castStringToSymEntry(strings, st, uint, errors), MsgType.NORMAL);
              }
              when "float64" {
                  return new MsgTuple(castStringToSymEntry(strings, st, real, errors), MsgType.NORMAL);
              }
              when "bool" {
                  return new MsgTuple(castStringToSymEntry(strings, st, bool, errors), MsgType.NORMAL);
              }
              otherwise {
                 var errorMsg = notImplementedError(pn,"str",":",targetDtype);
                 castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                 return new MsgTuple(errorMsg, MsgType.ERROR);
              }
          }
      }
      otherwise {
        var errorMsg = notImplementedError(pn,objtype);
        castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                      
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
  }

  proc registerMe() {
    use CommandMap;
    registerFunction("cast", castMsg, getModuleName());
  }
}
