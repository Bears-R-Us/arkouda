/* Array set operations
 includes intersection, union, xor, and diff

 currently, only performs operations with integer arrays 
 */

module ArraySetopsMsg
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedArray;
    use ServerErrorStrings;

    use ArraySetops;
    use Indexing;
    use RadixSortLSD;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;

    private config const logLevel = ServerConfig.logLevel;
    const asLogger = new Logger(logLevel);
    
    /*
    Parse, execute, and respond to a intersect1d message
    :arg reqMsg: request containing (cmd,name,name2,assume_unique)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (MsgTuple) response message
    */
    proc intersect1dMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, name2, assume_unique) = payload.splitMsgToTuple(3);
        var isUnique = stringtobool(assume_unique);
        
        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        var gEnt2: borrowed GenSymEntry = getGenericTypedArrayEntry(name2, st);
        
        // TODO look into this more, might be easier to move mem check into ArraySetOps
        // I'm not sure these are calculating the sort size correctly
        // because sometimes we sort both arrays so (gEnt_sortMem + gEnt2_sortMem)
        // and sometimes we sort the concat of the 2 arrays radixSortLSD_memEst(gEnt.size + gEnt2.size, max(gEnt.itemsize, gEnt2.itemsize), plan);
        const plan = makeRadixSortLSDPlan();
        const plan2 = makeRadixSortLSDPlan();
        var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize, plan = plan);
        var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize, plan = plan2);
        var intersect_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
        overMemLimit(intersect_maxMem);

        select(gEnt.dtype, gEnt2.dtype) {
          when (DType.Int64, DType.Int64) {
            var e = toSymEntry(gEnt,int);
            var f = toSymEntry(gEnt2,int);

            // it's unclear if plan or plan2 should be passed (should be equivalent rn), revisit along with mem estimates
            var aV = intersect1d(e.a, f.a, isUnique, plan = plan);
            st.addEntry(vname, new shared SymEntry(aV));

            repMsg = "created " + st.attrib(vname);
            asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.UInt64, DType.UInt64) {
            var e = toSymEntry(gEnt,uint);
            var f = toSymEntry(gEnt2,uint);

            // it's unclear if plan or plan2 should be passed (should be equivalent rn), revisit along with mem estimates
            var aV = intersect1d(e.a, f.a, isUnique, plan = plan);
            st.addEntry(vname, new shared SymEntry(aV));

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
    proc setxor1dMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, name2, assume_unique) = payload.splitMsgToTuple(3);
        var isUnique = stringtobool(assume_unique);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        var gEnt2: borrowed GenSymEntry = getGenericTypedArrayEntry(name2, st);

        // TODO look into this more, might be easier to move mem check into ArraySetOps
        // I'm not sure these are calculating the sort size correctly
        // because sometimes we sort both arrays so (gEnt_sortMem + gEnt2_sortMem)
        // and sometimes we sort the concat of the 2 arrays radixSortLSD_memEst(gEnt.size + gEnt2.size, max(gEnt.itemsize, gEnt2.itemsize), plan);
        const plan = makeRadixSortLSDPlan();
        const plan2 = makeRadixSortLSDPlan();
        var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize, plan = plan);
        var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize, plan = plan2);
        var xor_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
        overMemLimit(xor_maxMem);

        select(gEnt.dtype, gEnt2.dtype) {
          when (DType.Int64, DType.Int64) {
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2,int);
             
            // it's unclear if plan or plan2 should be passed (should be equivalent rn), revisit along with mem estimates
             var aV = setxor1d(e.a, f.a, isUnique, plan = plan);
             st.addEntry(vname, new shared SymEntry(aV));

             repMsg = "created " + st.attrib(vname);
             asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
             return new MsgTuple(repMsg, MsgType.NORMAL);
           }
           when (DType.UInt64, DType.UInt64) {
             var e = toSymEntry(gEnt,uint);
             var f = toSymEntry(gEnt2,uint);
             
            // it's unclear if plan or plan2 should be passed (should be equivalent rn), revisit along with mem estimates
             var aV = setxor1d(e.a, f.a, isUnique, plan = plan);
             st.addEntry(vname, new shared SymEntry(aV));

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
    proc setdiff1dMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, name2, assume_unique) = payload.splitMsgToTuple(3);
        var isUnique = stringtobool(assume_unique);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        var gEnt2: borrowed GenSymEntry = getGenericTypedArrayEntry(name2, st);

        // TODO look into this more, might be easier to move mem check into ArraySetOps
        // I'm not sure these are calculating the sort size correctly
        // because sometimes we sort both arrays so (gEnt_sortMem + gEnt2_sortMem)
        // and sometimes we sort the concat of the 2 arrays radixSortLSD_memEst(gEnt.size + gEnt2.size, max(gEnt.itemsize, gEnt2.itemsize), plan);
        const plan = makeRadixSortLSDPlan();
        const plan2 = makeRadixSortLSDPlan();
        var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize, plan = plan);
        var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize, plan = plan2);
        var diff_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
        overMemLimit(diff_maxMem);

        select(gEnt.dtype, gEnt2.dtype) {
          when (DType.Int64, DType.Int64) {
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2, int);
             
            // it's unclear if plan or plan2 should be passed (should be equivalent rn), revisit along with mem estimates
             var aV = setdiff1d(e.a, f.a, isUnique, plan = plan);
             st.addEntry(vname, new shared SymEntry(aV));

             var repMsg = "created " + st.attrib(vname);
             asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
             return new MsgTuple(repMsg, MsgType.NORMAL);
           }
           when (DType.UInt64, DType.UInt64) {
             var e = toSymEntry(gEnt,uint);
             var f = toSymEntry(gEnt2, uint);
             
            // it's unclear if plan or plan2 should be passed (should be equivalent rn), revisit along with mem estimates
             var aV = setdiff1d(e.a, f.a, isUnique, plan = plan);
             st.addEntry(vname, new shared SymEntry(aV));

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
    proc union1dMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      param pn = Reflection.getRoutineName();
      var repMsg: string; // response message
        // split request into fields
      var (name, name2) = payload.splitMsgToTuple(2);

      var vname = st.nextName();

      var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
      var gEnt2: borrowed GenSymEntry = getGenericTypedArrayEntry(name2, st);

      // TODO look into this more, might be easier to move mem check into ArraySetOps
      // I'm not sure these are calculating the sort size correctly
      // because sometimes we sort both arrays so (gEnt_sortMem + gEnt2_sortMem)
      // and sometimes we sort the concat of the 2 arrays radixSortLSD_memEst(gEnt.size + gEnt2.size, max(gEnt.itemsize, gEnt2.itemsize), plan);
      const plan = makeRadixSortLSDPlan();
      const plan2 = makeRadixSortLSDPlan();
      var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize, plan = plan);
      var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize, plan = plan2);
      var union_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
      overMemLimit(union_maxMem);

      select(gEnt.dtype, gEnt2.dtype) {
        when (DType.Int64, DType.Int64) {
           var e = toSymEntry(gEnt,int);
           var f = toSymEntry(gEnt2,int);

          // it's unclear if plan or plan2 should be passed (should be equivalent rn), revisit along with mem estimates
           var aV = union1d(e.a, f.a, plan = plan);
           st.addEntry(vname, new shared SymEntry(aV));

           var repMsg = "created " + st.attrib(vname);
           asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
           return new MsgTuple(repMsg, MsgType.NORMAL);
         }
         when (DType.UInt64, DType.UInt64) {
           var e = toSymEntry(gEnt,uint);
           var f = toSymEntry(gEnt2,uint);

          // it's unclear if plan or plan2 should be passed (should be equivalent rn), revisit along with mem estimates
           var aV = union1d(e.a, f.a, plan = plan);
           st.addEntry(vname, new shared SymEntry(aV));

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
    
    proc registerMe() {
      use CommandMap;
      registerFunction("intersect1d", intersect1dMsg, getModuleName());
      registerFunction("setdiff1d", setdiff1dMsg, getModuleName());
      registerFunction("setxor1d", setxor1dMsg, getModuleName());
      registerFunction("union1d", union1dMsg, getModuleName());
    }
}
