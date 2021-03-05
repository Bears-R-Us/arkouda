module SortMsg
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Sort only;
    use Reflection;
    use Errors;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use RadixSortLSD;
    use AryUtil;
    use Logging;
    use Message;
    
    const sortLogger = new Logger();
  
    if v {
        sortLogger.level = LogLevel.DEBUG;
    } else {
        sortLogger.level = LogLevel.INFO;
    }
  
    /* Sort the given pdarray using Radix Sort and
       return sorted keys as a block distributed array */
    proc sort(a: [?aD] ?t): [aD] t {
      var sorted: [aD] t = radixSortLSD_keys(a);
      return sorted;
    }

    /* sort takes pdarray and returns a sorted copy of the array */
    proc sortMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      param pn = Reflection.getRoutineName();
      var repMsg: string; // response message
      var (name) = payload.splitMsgToTuple(1);

      // get next symbol name
      var sortedName = st.nextName();

      var gEnt: borrowed GenSymEntry = st.lookup(name);

      // check and throw if over memory limit
      overMemLimit(((2 + 1) * gEnt.size * gEnt.itemsize)
                   + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
 
      sortLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                "cmd: %s name: %s sortedName: %s dtype: %t".format(
                 cmd, name, sortedName, gEnt.dtype));
      
      // Sort the input pda and create a new symbol entry for
      // the sorted pda.
      select (gEnt.dtype) {
          when (DType.Int64) {
              var e = toSymEntry(gEnt, int);
              var sorted = sort(e.a);
              st.addEntry(sortedName, new shared SymEntry(sorted));
          }// end when(DType.Int64)
          when (DType.Float64) {
              var e = toSymEntry(gEnt, real);
              var sorted = sort(e.a);
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
}// end module SortMsg
