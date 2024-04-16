module SetMsg {
  use Message;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use ServerConfig;
  use Logging;
  use ServerErrorStrings;
  use ServerErrors;
  use AryUtil;
  use CommAggregation;
  use RadixSortLSD;
  use Unique;

  use ArkoudaAryUtilCompat;

  use Reflection;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const sLogger = new Logger(logLevel, logChannel);

  @arkouda.registerND
  proc uniqueValuesMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          rname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc getUniqueVals(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd),
            eFlat = if nd == 1 then eIn.a else flatten(eIn.a);

      const eSorted = radixSortLSD_keys(eFlat);
      const eUnique = uniqueFromSorted(eSorted, needCounts=false);

      st.addEntry(rname, createSymEntry(eUnique));

      const repMsg = "created " + st.attrib(rname);
      sLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select gEnt.dtype {
      when DType.Int64 do return getUniqueVals(int);
      // when DType.UInt8 do return getUniqueVals(uint(8));
      when DType.UInt64 do return getUniqueVals(uint);
      when DType.Float64 do return getUniqueVals(real);
      when DType.Bool do return getUniqueVals(bool);
      otherwise {
        var errorMsg = notImplementedError(getRoutineName(),gEnt.dtype);
        sLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
  }

  @arkouda.registerND
  proc uniqueCountsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          uname = st.nextName(),
          cname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc getUniqueVals(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd),
            eFlat = if nd == 1 then eIn.a else flatten(eIn.a);

      const eSorted = radixSortLSD_keys(eFlat);
      const (eUnique, eCounts) = uniqueFromSorted(eSorted);

      st.addEntry(uname, createSymEntry(eUnique));
      st.addEntry(cname, createSymEntry(eCounts));

      const repMsg = "created " + st.attrib(uname) + "+created " + st.attrib(cname);
      sLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select gEnt.dtype {
      when DType.Int64 do return getUniqueVals(int);
      // when DType.UInt8 do return getUniqueVals(uint(8));
      when DType.UInt64 do return getUniqueVals(uint);
      when DType.Float64 do return getUniqueVals(real);
      when DType.Bool do return getUniqueVals(bool);
      otherwise {
        var errorMsg = notImplementedError(getRoutineName(),gEnt.dtype);
        sLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
  }

  @arkouda.registerND
  proc uniqueInverseMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          uname = st.nextName(),
          iname = st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc getUniqueVals(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd),
            eFlat = if nd == 1 then eIn.a else flatten(eIn.a);

      const (eUnique, _, inv) = uniqueSortWithInverse(eFlat);
      st.addEntry(uname, createSymEntry(eUnique));
      st.addEntry(iname, createSymEntry(if nd == 1 then inv else unflatten(inv, eIn.a.shape)));

      const repMsg = "created " + st.attrib(uname) + "+created " + st.attrib(iname);
      sLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select gEnt.dtype {
      when DType.Int64 do return getUniqueVals(int);
      // when DType.UInt8 do return getUniqueVals(uint(8));
      when DType.UInt64 do return getUniqueVals(uint);
      when DType.Float64 do return getUniqueVals(real);
      when DType.Bool do return getUniqueVals(bool);
      otherwise {
        var errorMsg = notImplementedError(getRoutineName(),gEnt.dtype);
        sLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
  }

  @arkouda.registerND
  proc uniqueAllMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          rnames = for 0..<4 do st.nextName();

    var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

    proc getUniqueVals(type t): MsgTuple throws {
      const eIn = toSymEntry(gEnt, t, nd),
            eFlat = if nd == 1 then eIn.a else flatten(eIn.a);

      const (eUnique, eCounts, inv, eIndices) = uniqueSortWithInverse(eFlat, needIndices=true);
      st.addEntry(rnames[0], createSymEntry(eUnique));
      st.addEntry(rnames[1], createSymEntry(eIndices));
      st.addEntry(rnames[2], createSymEntry(if nd == 1 then inv else unflatten(inv, eIn.a.shape)));
      st.addEntry(rnames[3], createSymEntry(eCounts));

      const repMsg = try! "+".join([rn in rnames] "created " + st.attrib(rn));
      sLogger.info(getModuleName(),pn,getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    select gEnt.dtype {
      when DType.Int64 do return getUniqueVals(int);
      // when DType.UInt8 do return getUniqueVals(uint(8));
      when DType.UInt64 do return getUniqueVals(uint);
      when DType.Float64 do return getUniqueVals(real);
      when DType.Bool do return getUniqueVals(bool);
      otherwise {
        var errorMsg = notImplementedError(getRoutineName(),gEnt.dtype);
        sLogger.error(getModuleName(),pn,getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
  }
}
