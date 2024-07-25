module ZarrMsg {
  use IO;
  use ServerErrors, ServerConfig;
  use FileIO;
  use FileSystem;
  use GenSymIO;
  use List;
  use Logging;
  use Message;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use NumPyDType;
  use Sort;
  use CommAggregation;
  use AryUtil;
  use CTypes;
  use Zarr;
  use ServerConfig;

  use Reflection;
  use ServerErrors;
  use ServerErrorStrings;
  use SegmentedString;

  use Map;
  use Math;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const zarrLogger = new Logger(logLevel, logChannel);

  @arkouda.registerND(cmd_prefix="readAllZarr")
  proc readAllZarrMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    const storePath = msgArgs.getValueOf("store_path"), dtype = str2dtype(msgArgs.getValueOf("dtype"));
    const rname = st.nextName();
    select dtype {
      when DType.Float64 {
        var ar1 = readZarrArray(storePath, real, nd);
        var ar2 = makeDistArray((...ar1.shape), real);
        ar2 = ar1;
        st.addEntry(rname, createSymEntry(ar2));
      }
      when DType.Float32 {
        var ar1 = readZarrArray(storePath, real(32), nd);
        var ar2 = makeDistArray((...ar1.shape), real(32));
        ar2 = ar1;
        st.addEntry(rname, createSymEntry(ar2));
      }
      when DType.Int64 {
        var ar1 = readZarrArray(storePath, int(64), nd);
        var ar2 = makeDistArray((...ar1.shape), int(64));
        ar2 = ar1;
        st.addEntry(rname, createSymEntry(ar2));
      }
      when DType.Int32 {
        var ar1 = readZarrArray(storePath, int(32), nd);
        var ar2 = makeDistArray((...ar1.shape), int(32));
        ar2 = ar1;
        st.addEntry(rname, createSymEntry(ar2));
      }
    }

    const repMsg = "created  " + st.attrib(rname);
    zarrLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  @arkouda.registerND(cmd_prefix="writeAllZarr")
  proc writeAllZarrMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    zarrLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "writing zarr array");
    const storePath = msgArgs.getValueOf("store_path");
    const name = msgArgs.getValueOf("arr");
    const chunkShape = msgArgs.get("chunk_shape").getTuple(nd);
    
    zarrLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "%s".format(name));
    var gAr1: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    var failed = false;
    try {
      select gAr1.dtype {
        when DType.Float64 {
          var ar1 = toSymEntry(gAr1,real,nd);
          writeZarrArray(storePath, ar1.a, chunkShape);
        }
        when DType.Float32 {
          var ar1 = toSymEntry(gAr1,real(32),nd);
          writeZarrArray(storePath, ar1.a, chunkShape);
        }
        when DType.Int64 {
          var ar1 = toSymEntry(gAr1,int,nd);
          writeZarrArray(storePath, ar1.a, chunkShape);
        }
        when DType.Int32 {
          var ar1 = toSymEntry(gAr1,int(32),nd);
          writeZarrArray(storePath, ar1.a, chunkShape);
        }
        otherwise do failed=true;
      }
    } catch e {
      throw getErrorWithContext(
                     msg="%s".format(e),
                     getLineNumber(),
                     getRoutineName(),
                     getModuleName(),
                     errorClass="IOError");
    }
    if failed {
      throw getErrorWithContext(
                     msg="unsupported dtype %s".format(gAr1.dtype),
                     getLineNumber(),
                     getRoutineName(),
                     getModuleName(),
                     errorClass="TypeError");
    }
    // if verbose print result
    zarrLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "wrote the pdarray %s".format(st.attrib(name)));

    const repMsg = "wrote " + st.attrib(name);
    zarrLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }
  

  
  
}
