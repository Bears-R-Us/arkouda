module SortMsg
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Sort only;
    use Reflection;
    use ServerErrors;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use RadixSortLSD;
    use AryUtil;
    use Logging;
    use Message;
    private use ArgSortMsg;

    private config const logLevel = ServerConfig.logLevel;
    const sortLogger = new Logger(logLevel);

    /* Sort the given pdarray using Radix Sort and
       return sorted keys as a block distributed array */
    proc sort(a: [?aD] ?t): [aD] t throws {
      var sorted: [aD] t = radixSortLSD_keys(a);
      return sorted;
    }

    /* sort takes pdarray and returns a sorted copy of the array */
    proc sortMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      param pn = Reflection.getRoutineName();
      var repMsg: string; // response message
      var (algoName, name) = payload.splitMsgToTuple(2);
      var algorithm: SortingAlgorithm = defaultSortAlgorithm;
      if algoName != "" {
        try {
          algorithm = algoName: SortingAlgorithm;
        } catch {
          throw getErrorWithContext(
                                    msg="Unrecognized sorting algorithm: %s".format(algoName),
                                    lineNumber=getLineNumber(),
                                    routineName=getRoutineName(),
                                    moduleName=getModuleName(),
                                    errorClass="NotImplementedError"
                                    );
        }
      }
      // get next symbol name
      var sortedName = st.nextName();

      var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

      // check and throw if over memory limit
      overMemLimit(radixSortLSD_keys_memEst(gEnt.size,  gEnt.itemsize));
 
      sortLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "cmd: %s name: %s sortedName: %s dtype: %t".format(
                 cmd, name, sortedName, gEnt.dtype));

      proc doSort(a: [?D] ?t) throws {
        select algorithm {
          when SortingAlgorithm.TwoArrayRadixSort {
            var b: [D] t = a;
            Sort.TwoArrayRadixSort.twoArrayRadixSort(b, comparator=myDefaultComparator);
            return b;
          }
          when SortingAlgorithm.RadixSortLSD {
            return radixSortLSD_keys(a);
          }
          otherwise {
            throw getErrorWithContext(
                                      msg="Unrecognized sorting algorithm: %s".format(algorithm:string),
                                      lineNumber=getLineNumber(),
                                      routineName=getRoutineName(),
                                      moduleName=getModuleName(),
                                      errorClass="NotImplementedError"
                                      );
          }
        }
      }
      // Sort the input pda and create a new symbol entry for
      // the sorted pda.
      select (gEnt.dtype) {
          when (DType.Int64) {
              var e = toSymEntry(gEnt, int);
              var sorted = doSort(e.a);
              st.addEntry(sortedName, new shared SymEntry(sorted));
          }// end when(DType.Int64)
          when (DType.UInt64) {
              var e = toSymEntry(gEnt, uint);
              var sorted = doSort(e.a);
              st.addEntry(sortedName, new shared SymEntry(sorted));
          }// end when(DType.UInt64)
          when (DType.Float64) {
              var e = toSymEntry(gEnt, real);
              var sorted = doSort(e.a);
              st.addEntry(sortedName, new shared SymEntry(sorted));
          }// end when(DType.Float64)
          otherwise {
              var errorMsg = notImplementedError(pn,gEnt.dtype);
              sortLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                                     errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
          }            
      }// end select(gEnt.dtype)

      repMsg = "created " + st.attrib(sortedName);
      sortLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }// end sortMsg()

    use CommandMap;
    registerFunction("sort", sortMsg, getModuleName());
}// end module SortMsg
