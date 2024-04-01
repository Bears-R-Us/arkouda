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

  // TODO: put this in AryUtil or some other common module after merging with #3056
  private proc unflatten(const ref aFlat: [?d] ?t, shape: ?N*int): [] t throws {
    var unflat = makeDistArray((...shape), t);
    const lastRank = unflat.domain.dim(N-1);

    // iterate over each slice of the output array along the last dimension
    // and copy the data from the corresponding slice of the flat array
    forall idx in domOffAxis(unflat.domain, N-1) with (const ord = new orderer(unflat.domain.shape)) {
      var idxTup: (N-1)*int;
      for i in 0..<(N-1) do idxTup[i] = idx[i];
      const rrSlice = ((...idxTup), lastRank);

      const low = ((...idxTup), lastRank.low),
            high = ((...idxTup), lastRank.high),
            flatSlice = ord.indexToOrder(low)..ord.indexToOrder(high);

      unflat[(...rrSlice)] = aFlat[flatSlice];
    }

    return unflat;
  }

  // TODO: put this in AryUtil or some other common module after merging with #3056
  private proc flatten(const ref a: [?d] ?t): [] t throws
    where a.rank > 1
  {
    var flat = makeDistArray({0..<d.size}, t);
    const rankLast = d.dim(d.rank-1);

    // iterate over each slice of the input array along the last dimension
    // and copy the data into the corresponding slice of the flat array
    forall idx in domOffAxis(d, d.rank-1) with (const ord = new orderer(d.shape)) {
      var idxTup: (d.rank-1)*int;
      for i in 0..<(d.rank-1) do idxTup[i] = idx[i];
      const rrSlice = ((...idxTup), rankLast);

      const low = ((...idxTup), rankLast.low),
            high = ((...idxTup), rankLast.high),
            flatSlice = ord.indexToOrder(low)..ord.indexToOrder(high);

      flat[flatSlice] = a[(...rrSlice)];
    }

    return flat;
  }

  record orderer {
    param rank: int;
    const accumRankSizes: [0..<rank] int;

    proc init(shape: ?N*int) {
      this.rank = N;
      const sizesRev = [i in 0..<N] shape[N - i - 1];
      this.accumRankSizes = * scan sizesRev / sizesRev;
    }

    // index -> order for the input array's indices
    // e.g., order = k + (nz * j) + (nz * ny * i)
    inline proc indexToOrder(idx: rank*int): int {
      var order = 0;
      for param i in 0..<rank do order += idx[i] * accumRankSizes[rank - i - 1];
      return order;
    }
  }
}
