module BroadcastMsg {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Errors;
  use Reflection;
  use Broadcast;
  use ServerConfig;
  use Logging;
  use Message;
  
  const bmLogger = new Logger();

  if v {
        bmLogger.level = LogLevel.DEBUG;
  } else {
        bmLogger.level = LogLevel.INFO;
  }
  
  /* 
   * Broadcast a value per segment of a segmented array to the
   * full size of the array, optionally applying a permutation
   * to return the result in the order of the original array.
   */
  proc broadcastMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var (permName, segName, valName, usePermStr, sizeStr) = payload.splitMsgToTuple(5);
    const size = sizeStr: int;
    // Segments must be an int64 array
    const gs = st.lookup(segName);
    if gs.dtype != DType.Int64 {
      throw new owned ErrorWithContext("Segments array must have dtype int64",
                                       getLineNumber(),
                                       getRoutineName(),
                                       getModuleName(),
                                       "TypeError");
    }
    const segs = toSymEntry(gs, int);
    // Check that values exists (can be any dtype)
    const gv = st.lookup(valName);
    // Name of result array
    const rname = st.nextName();
    // This operation has two modes: one uses a permutation to reorder the answer,
    // while the other does not
    const usePerm: bool = usePermStr.toLower() == 'true';
    if usePerm {
      // If using a permutation, the array must be int64 and same size as the size parameter
      const gp = st.lookup(permName);
      if gp.dtype != DType.Int64 {
        throw new owned ErrorWithContext("Permutation array must have dtype int64",
                                         getLineNumber(),
                                         getRoutineName(),
                                         getModuleName(),
                                         "TypeError");
      }
      if gp.size != size {
        throw new owned ErrorWithContext("Requested result size must match permutation array size",
                                         getLineNumber(),
                                         getRoutineName(),
                                         getModuleName(),
                                         "ValueError");
      }
      const perm = toSymEntry(gp, int);
      // Select on dtype of values
      select gv.dtype {
        when DType.Int64 {
          const vals = toSymEntry(gv, int);
          var res = st.addEntry(rname, size, int);
          res.a = broadcast(perm.a, segs.a, vals.a);
        }
        when DType.Float64 {
          const vals = toSymEntry(gv, real);
          var res = st.addEntry(rname, size, real);
          res.a = broadcast(perm.a, segs.a, vals.a);
        }
        when DType.Bool {
          const vals = toSymEntry(gv, bool);
          var res = st.addEntry(rname, size, bool);
          res.a = broadcast(perm.a, segs.a, vals.a);
        }
        otherwise {
          throw new owned ErrorWithContext("Values array has unsupported dtype %s".format(gv.dtype:string),
                                           getLineNumber(),
                                           getRoutineName(),
                                           getModuleName(),
                                           "TypeError");
        }
      }
    } else {
      // If not using permutation, ignore perm array
      select gv.dtype {
        when DType.Int64 {
          const vals = toSymEntry(gv, int);
          var res = st.addEntry(rname, size, int);
          res.a = broadcast(segs.a, vals.a, size);
        }
        when DType.Float64 {
          const vals = toSymEntry(gv, real);
          var res = st.addEntry(rname, size, real);
          res.a = broadcast(segs.a, vals.a, size);
        }
        when DType.Bool {
          const vals = toSymEntry(gv, bool);
          var res = st.addEntry(rname, size, bool);
          res.a = broadcast(segs.a, vals.a, size);
        }
        otherwise {
          throw new owned ErrorWithContext("Values array has unsupported dtype %s".format(gv.dtype:string),
                                           getLineNumber(),
                                           getRoutineName(),
                                           getModuleName(),
                                           "TypeError");
        }
      }
    }
    var repMsg = "created " + st.attrib(rname); 
    bmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);    
  }
}
