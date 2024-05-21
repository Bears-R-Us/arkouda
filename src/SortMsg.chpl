module SortMsg
{
    use ServerConfig;

    use ArkoudaTimeCompat as Time;
    use Math only;
    use Sort only;
    use Search only;
    use Reflection;
    use ServerErrors;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use RadixSortLSD;
    use AryUtil; use ArkoudaAryUtilCompat;
    use Logging;
    use Message;
    private use ArgSortMsg;
    use ArkoudaSortCompat;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const sortLogger = new Logger(logLevel, logChannel);

    /* Sort the given pdarray using Radix Sort and
       return sorted keys as a block distributed array */
    proc sort(a: [?aD] ?t): [aD] t throws {
      var sorted: [aD] t = radixSortLSD_keys(a);
      return sorted;
    }

    /* sort takes pdarray and returns a sorted copy of the array */
    @arkouda.registerND
    proc sortMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
      param pn = Reflection.getRoutineName();
      const algoName = msgArgs.getValueOf("alg"),
            name = msgArgs.getValueOf("array"),
            axis = msgArgs.get("axis").getIntValue(),
            rname = st.nextName();

      var algorithm: SortingAlgorithm = defaultSortAlgorithm;
      if algoName != "" {
        try {
          algorithm = algoName: SortingAlgorithm;
        } catch {
          throw getErrorWithContext(
            msg="Unrecognized sorting algorithm: %s".doFormat(algoName),
            lineNumber=getLineNumber(),
            pn,
            moduleName=getModuleName(),
            errorClass="NotImplementedError"
          );
        }
      }

      var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

      sortLogger.debug(
        getModuleName(),pn,getLineNumber(),
        "cmd: %s, name: %s, sortedName: %s, dtype: %?, nd: %i, axis: %i".doFormat(
          cmd, name, rname, gEnt.dtype, nd, axis
        )
      );

      proc doSort(type t): MsgTuple throws
        where nd == 1
      {
        overMemLimit(radixSortLSD_keys_memEst(gEnt.size, gEnt.itemsize));

        const e = toSymEntry(gEnt, t);

        if algorithm == SortingAlgorithm.TwoArrayRadixSort {
          var sorted = makeDistArray(e.a);
          ArkoudaSortCompat.twoArrayRadixSort(sorted, comparator=myDefaultComparator);
          st.addEntry(rname, createSymEntry(sorted));
        } else {
          var sorted = radixSortLSD_keys(e.a);
          st.addEntry(rname, createSymEntry(sorted));
        }

        const repMsg = "created " + st.attrib(rname);
        sortLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
      }

      proc doSort(type t): MsgTuple throws
        where nd > 1
      {
        const e = toSymEntry(gEnt, t, nd),
              DD = domOffAxis(e.a.domain, axis);
        var sorted = makeDistArray((...e.a.domain.shape), t);

        if algorithm == SortingAlgorithm.TwoArrayRadixSort {
          for idx in DD {
            // make a copy of the array along the slice corresponding to idx
            // TODO: create a twoArrayRadixSort that operates on a slice of the array
            // in place instead of requiring the copy in/out
            var slice = makeDistArray(e.a.domain.dim(axis).size, t);
            forall i in e.a.domain.dim(axis) with (var perpIdx = idx) {
              perpIdx[axis] = i;
              slice[i] = e.a[perpIdx];
            }

            ArkoudaSortCompat.twoArrayRadixSort(slice, comparator=myDefaultComparator);

            forall i in e.a.domain.dim(axis) with (var perpIdx = idx) {
              perpIdx[axis] = i;
              sorted[perpIdx] = slice[i];
            }
          }
        } else {
          // TODO: make a version of radixSortLSD_keys that does the sort on
          // slices of `e.a` directly instead of requiring a copy for each slice
          for idx in DD {
            const sliceDom = domOnAxis(e.a.domain, idx, axis),
                  sliced1D = removeDegenRanks(e.a[sliceDom], 1),
                  sliceSorted = radixSortLSD_keys(sliced1D);

            forall i in sliceDom do sorted[i] = sliceSorted[i[axis]];
          }
        }

        st.addEntry(rname, createSymEntry(sorted));
        const repMsg = "created " + st.attrib(rname);
        sortLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
      }

      select gEnt.dtype {
        when DType.Int64 do return doSort(int);
        when DType.UInt64 do return doSort(uint);
        when DType.Float64 do return doSort(real);
        otherwise {
          var errorMsg = notImplementedError(pn,gEnt.dtype);
          sortLogger.error(getModuleName(),pn,getLineNumber(), errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
    }

    // https://data-apis.org/array-api/latest/API_specification/generated/array_api.searchsorted.html#array_api.searchsorted
    @arkouda.registerND
    proc searchSortedMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
      param pn = Reflection.getRoutineName();
      const x1 = msgArgs.getValueOf("x1"),
            x2 = msgArgs.getValueOf("x2"),
            side = msgArgs.getValueOf("side"),
            rname = st.nextName();

      var gEntX1: borrowed GenSymEntry = getGenericTypedArrayEntry(x1, st),
          gEntX2: borrowed GenSymEntry = getGenericTypedArrayEntry(x2, st);

      if side != "left" && side != "right" {
        throw getErrorWithContext(
          msg="Unrecognized side: %s".doFormat(side),
          lineNumber=getLineNumber(),
          pn,
          moduleName=getModuleName(),
          errorClass="NotImplementedError"
        );
      }

      // TODO: add support for Float32
      if gEntX1.dtype != DType.Float64 || gEntX2.dtype != DType.Float64 {
        throw getErrorWithContext(
          msg="searchsorted only supports Float64 arrays",
          lineNumber=getLineNumber(),
          pn,
          moduleName=getModuleName(),
          errorClass="NotImplementedError"
        );
      }

      sortLogger.debug(
        getModuleName(),pn,getLineNumber(),
        "cmd: %s, x1: %s, x2: %s, side: %s, rname: %s, dtype: %?, nd: %i".doFormat(
          cmd, x1, x2, side, rname, gEntX1.dtype, nd
        )
      );

      const e1 = toSymEntry(gEntX1, real, 1),
            e2 = toSymEntry(gEntX2, real, nd);
      var ret = makeDistArray((...e2.a.domain.shape), int);

      proc doSearch(const ref a1: [] real, const ref a2: [?d] real, cmp) {
        forall idx in ret.domain {
          const (_, i) = Search.binarySearch(a1, a2[idx], cmp);
          ret[idx] = i;
        }
      }

      select side {
        when "left" do doSearch(e1.a, e2.a, new leftCmp());
        when "right" do doSearch(e1.a, e2.a, new rightCmp());
        otherwise do halt("unreachable");
      }

      st.addEntry(rname, createSymEntry(ret));
      const repMsg = "created " + st.attrib(rname);
      sortLogger.debug(getModuleName(),pn,getLineNumber(),repMsg);

      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    record leftCmp {
      proc compare(a: real, b: real): int {
        if a < b then return -1;
        else return 1;
      }
    }

    record rightCmp {
      proc compare(a: real, b: real): int {
        if a <= b then return -1;
        else return 1;
      }
    }

}// end module SortMsg
