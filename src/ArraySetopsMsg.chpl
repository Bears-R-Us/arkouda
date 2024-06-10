/* Array set operations
 includes intersection, union, xor, and diff

 currently, only performs operations with integer arrays 
 */

module ArraySetopsMsg
{
    use ServerConfig;

    use Time;
    use Math only;
    use Reflection only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedString;
    use ServerErrorStrings;

    use ArraySetops;
    use Indexing;
    use RadixSortLSD;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const asLogger = new Logger(logLevel, logChannel);
    
    /*
    Parse, execute, and respond to a intersect1d message
    :arg reqMsg: request containing (cmd,name,name2,assume_unique)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (MsgTuple) response message
    */
    proc intersect1dMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var isUnique = msgArgs.get("assume_unique").getBoolValue();
        
        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("arg1"), st);
        var gEnt2: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("arg2"), st);
        
        var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
        var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
        var intersect_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
        overMemLimit(intersect_maxMem);

        select(gEnt.dtype, gEnt2.dtype) {
          when (DType.Int64, DType.Int64) {
            var e = toSymEntry(gEnt,int);
            var f = toSymEntry(gEnt2,int);

            var aV = intersect1d(e.a, f.a, isUnique);
            st.addEntry(vname, createSymEntry(aV));

            repMsg = "created " + st.attrib(vname);
            asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.UInt64, DType.UInt64) {
            var e = toSymEntry(gEnt,uint);
            var f = toSymEntry(gEnt2,uint);

            var aV = intersect1d(e.a, f.a, isUnique);
            st.addEntry(vname, createSymEntry(aV));

            repMsg = "created " + st.attrib(vname);
            asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          otherwise {
            var errorMsg = notImplementedError("intersect1d",gEnt.dtype);
            asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);           
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
    }

    /*
    Parse, execute, and respond to a setxor1d message
    :arg reqMsg: request containing (cmd,name,name2,assume_unique)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (MsgTuple) response message
    */
    proc setxor1dMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var isUnique = msgArgs.get("assume_unique").getBoolValue();

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("arg1"), st);
        var gEnt2: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("arg2"), st);

        var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
        var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
        var xor_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
        overMemLimit(xor_maxMem);

        select(gEnt.dtype, gEnt2.dtype) {
          when (DType.Int64, DType.Int64) {
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2,int);
             
             var aV = setxor1d(e.a, f.a, isUnique);
             st.addEntry(vname, createSymEntry(aV));

             repMsg = "created " + st.attrib(vname);
             asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
             return new MsgTuple(repMsg, MsgType.NORMAL);
           }
           when (DType.UInt64, DType.UInt64) {
             var e = toSymEntry(gEnt,uint);
             var f = toSymEntry(gEnt2,uint);
             
             var aV = setxor1d(e.a, f.a, isUnique);
             st.addEntry(vname, createSymEntry(aV));

             repMsg = "created " + st.attrib(vname);
             asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
             return new MsgTuple(repMsg, MsgType.NORMAL);
           }
           otherwise {
               var errorMsg = notImplementedError("setxor1d",gEnt.dtype);
               asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                  
               return new MsgTuple(errorMsg, MsgType.ERROR);
           }
        }
    }

    /*
    Parse, execute, and respond to a setdiff1d message
    :arg reqMsg: request containing (cmd,name,name2,assume_unique)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (MsgTuple) response message
    */
    proc setdiff1dMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var isUnique = msgArgs.get("assume_unique").getBoolValue();

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("arg1"), st);
        var gEnt2: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("arg2"), st);

        var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
        var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
        var diff_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
        overMemLimit(diff_maxMem);

        select(gEnt.dtype, gEnt2.dtype) {
          when (DType.Int64, DType.Int64) {
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2, int);
             
             var aV = setdiff1d(e.a, f.a, isUnique);
             st.addEntry(vname, createSymEntry(aV));

             var repMsg = "created " + st.attrib(vname);
             asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
             return new MsgTuple(repMsg, MsgType.NORMAL);
           }
           when (DType.UInt64, DType.UInt64) {
             var e = toSymEntry(gEnt,uint);
             var f = toSymEntry(gEnt2, uint);
             
             var aV = setdiff1d(e.a, f.a, isUnique);
             st.addEntry(vname, createSymEntry(aV));

             var repMsg = "created " + st.attrib(vname);
             asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
             return new MsgTuple(repMsg, MsgType.NORMAL);
           }
           otherwise {
               var errorMsg = notImplementedError("setdiff1d",gEnt.dtype);
               asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                 
               return new MsgTuple(errorMsg, MsgType.ERROR);           
           }
        }
    }

    /*
    Parse, execute, and respond to a union1d message
    :arg reqMsg: request containing (cmd,name,name2,assume_unique)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (MsgTuple) response message
    */
    proc union1dMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      param pn = Reflection.getRoutineName();
      var repMsg: string; // response message

      var vname = st.nextName();

      var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("arg1"), st);
      var gEnt2: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("arg2"), st);
      
      var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
      var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
      var union_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
      overMemLimit(union_maxMem);

      select(gEnt.dtype, gEnt2.dtype) {
        when (DType.Int64, DType.Int64) {
           var e = toSymEntry(gEnt,int);
           var f = toSymEntry(gEnt2,int);

           var aV = union1d(e.a, f.a);
           st.addEntry(vname, createSymEntry(aV));

           var repMsg = "created " + st.attrib(vname);
           asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
           return new MsgTuple(repMsg, MsgType.NORMAL);
         }
         when (DType.UInt64, DType.UInt64) {
           var e = toSymEntry(gEnt,uint);
           var f = toSymEntry(gEnt2,uint);

           var aV = union1d(e.a, f.a);
           st.addEntry(vname, createSymEntry(aV));

           var repMsg = "created " + st.attrib(vname);
           asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
           return new MsgTuple(repMsg, MsgType.NORMAL);
         }
         otherwise {
             var errorMsg = notImplementedError("newUnion1d",gEnt.dtype);
             asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                   
             return new MsgTuple(errorMsg, MsgType.ERROR);              
         }
      }
    }

    proc stringtobool(str: string): bool throws {
        if str == "True" then return true;
        else if str == "False" then return false;
        throw getErrorWithContext(
            msg="message: assume_unique must be of type bool",
            lineNumber=getLineNumber(),
            routineName=getRoutineName(),
            moduleName=getModuleName(),
            errorClass="ErrorWithContext");
    }

    proc sparseSumHelpMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      param pn = Reflection.getRoutineName();
      var repMsg: string; // response message

      var iname = st.nextName();
      var vname = st.nextName();

      const gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("idx1"), st);
      const gEnt2: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("idx2"), st);
      const gEnt3: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("val1"), st);
      const gEnt4: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("val2"), st);
      const doMerge = msgArgs.get("merge").getBoolValue();
      const percentTransferLimit = msgArgs.get("percent_transfer_limit").getIntValue();

      const gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
      const gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
      const union_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
      overMemLimit(union_maxMem);

      select(gEnt.dtype, gEnt2.dtype, gEnt3.dtype, gEnt4.dtype) {
        when (DType.Int64, DType.Int64, DType.Int64, DType.Int64) {
          const e = toSymEntry(gEnt,int);
          const f = toSymEntry(gEnt2,int);
          const g = toSymEntry(gEnt3,int);
          const h = toSymEntry(gEnt4,int);
          const ref ea = e.a;
          const ref fa = f.a;
          const ref ga = g.a;
          const ref ha = h.a;

          const (retIdx, retVals) = sparseSumHelper(ea, fa, ga, ha, doMerge, percentTransferLimit);
          st.addEntry(iname, createSymEntry(retIdx));
          st.addEntry(vname, createSymEntry(retVals));

          const repMsg = "created " + st.attrib(iname) + "+created " + st.attrib(vname);
          asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
          return new MsgTuple(repMsg, MsgType.NORMAL);
        }
        when (DType.Int64, DType.Int64, DType.UInt64, DType.UInt64) {
          const e = toSymEntry(gEnt,int);
          const f = toSymEntry(gEnt2,int);
          const g = toSymEntry(gEnt3,uint);
          const h = toSymEntry(gEnt4,uint);
          const ref ea = e.a;
          const ref fa = f.a;
          const ref ga = g.a;
          const ref ha = h.a;

          const (retIdx, retVals) = sparseSumHelper(ea, fa, ga, ha, doMerge, percentTransferLimit);
          st.addEntry(iname, createSymEntry(retIdx));
          st.addEntry(vname, createSymEntry(retVals));

          const repMsg = "created " + st.attrib(iname) + "+created " + st.attrib(vname);
          asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
          return new MsgTuple(repMsg, MsgType.NORMAL);
        }
        otherwise {
          var errorMsg = notImplementedError("sparseSumHelper",gEnt.dtype);
          asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
    }
    
    use CommandMap;
    registerFunction("intersect1d", intersect1dMsg, getModuleName());
    registerFunction("setdiff1d", setdiff1dMsg, getModuleName());
    registerFunction("setxor1d", setxor1dMsg, getModuleName());
    registerFunction("union1d", union1dMsg, getModuleName());
    registerFunction("sparseSumHelp", sparseSumHelpMsg, getModuleName());
}
