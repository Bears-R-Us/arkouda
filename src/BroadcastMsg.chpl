module BroadcastMsg {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Errors;
  use Reflection;
  use Broadcast;

  proc broadcastMsg(cmd: string, payload: bytes, st: borrowed SymTab) throws {
    var (permName, segName, valName, usePermStr, sizeStr) = payload.decode().splitMsgToTuple(5);
    const size = sizeStr: int;
    const gs = st.lookup(segName);
    if gs.dtype != DType.Int64 {
      throw new owned ErrorWithContext("Segments array must have dtype int64",
                                       getLineNumber(),
                                       getRoutineName(),
                                       getModuleName(),
                                       "TypeError");
    }
    const segs = toSymEntry(gs, int);
    const gv = st.lookup(valName);
    const rname = st.nextName();
    const usePerm: bool = usePermStr.toLower() == 'true';
    if usePerm {
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
    return "created " + st.attrib(rname); 
  }

}