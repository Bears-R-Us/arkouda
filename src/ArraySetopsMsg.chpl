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
    use Unique;
    use Indexing;
    use RadixSortLSD;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use CommAggregation;

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
        
        var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
        var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
        var intersect_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
        overMemLimit(intersect_maxMem);

        select(gEnt.dtype, gEnt2.dtype) {
          when (DType.Int64, DType.Int64) {
            var e = toSymEntry(gEnt,int);
            var f = toSymEntry(gEnt2,int);

            var aV = intersect1d(e.a, f.a, isUnique);
            st.addEntry(vname, new shared SymEntry(aV));

            repMsg = "created " + st.attrib(vname);
            asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.UInt64, DType.UInt64) {
            var e = toSymEntry(gEnt,uint);
            var f = toSymEntry(gEnt2,uint);

            var aV = intersect1d(e.a, f.a, isUnique);
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

        var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
        var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
        var xor_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
        overMemLimit(xor_maxMem);

        select(gEnt.dtype, gEnt2.dtype) {
          when (DType.Int64, DType.Int64) {
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2,int);
             
             var aV = setxor1d(e.a, f.a, isUnique);
             st.addEntry(vname, new shared SymEntry(aV));

             repMsg = "created " + st.attrib(vname);
             asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
             return new MsgTuple(repMsg, MsgType.NORMAL);
           }
           when (DType.UInt64, DType.UInt64) {
             var e = toSymEntry(gEnt,uint);
             var f = toSymEntry(gEnt2,uint);
             
             var aV = setxor1d(e.a, f.a, isUnique);
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

        var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
        var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
        var diff_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
        overMemLimit(diff_maxMem);

        select(gEnt.dtype, gEnt2.dtype) {
          when (DType.Int64, DType.Int64) {
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2, int);
             
             var aV = setdiff1d(e.a, f.a, isUnique);
             st.addEntry(vname, new shared SymEntry(aV));

             var repMsg = "created " + st.attrib(vname);
             asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
             return new MsgTuple(repMsg, MsgType.NORMAL);
           }
           when (DType.UInt64, DType.UInt64) {
             var e = toSymEntry(gEnt,uint);
             var f = toSymEntry(gEnt2, uint);
             
             var aV = setdiff1d(e.a, f.a, isUnique);
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
      
      var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
      var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
      var union_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
      overMemLimit(union_maxMem);

      select(gEnt.dtype, gEnt2.dtype) {
        when (DType.Int64, DType.Int64) {
           var e = toSymEntry(gEnt,int);
           var f = toSymEntry(gEnt2,int);

           var aV = union1d(e.a, f.a);
           st.addEntry(vname, new shared SymEntry(aV));

           var repMsg = "created " + st.attrib(vname);
           asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
           return new MsgTuple(repMsg, MsgType.NORMAL);
         }
         when (DType.UInt64, DType.UInt64) {
           var e = toSymEntry(gEnt,uint);
           var f = toSymEntry(gEnt2,uint);

           var aV = union1d(e.a, f.a);
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

    proc union1d_multiMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      param pn = Reflection.getRoutineName();
      var repMsg: string; // response message
        // split request into fields
      var (seg1_name, vals1_name, s1_str, seg2_name, vals2_name, s2_str) = payload.splitMsgToTuple(6);

      var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(seg1_name, st);
      var gEnt2: borrowed GenSymEntry = getGenericTypedArrayEntry(vals1_name, st);
      var gEnt3: borrowed GenSymEntry = getGenericTypedArrayEntry(seg2_name, st);
      var gEnt4: borrowed GenSymEntry = getGenericTypedArrayEntry(vals2_name, st);

      // verify that expected integer values can be cast
      var size1: int;
      var size2: int;
      try{
        size1 = s1_str: int;
        size2 = s2_str: int;
      }
      catch {
        var errorMsg = "Count could not be interpretted as an int1: %s, int2: %s)".format(s1_str, s2_str);
        asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        throw new owned IllegalArgumentError(errorMsg);
      }

      var segments1 = toSymEntry(gEnt,int);
      var segments2 = toSymEntry(gEnt3,int);

      // set segment lengths
      var lens1: [segments1.aD] int;
      var lens2: [segments2.aD] int;
      var high = segments1.aD.high;
      var low = segments1.aD.low;
      var m1: int;
      var m2: int;
      for (i, s1, l1, s2, l2) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2){
        if i == high {
          l1 = size1 - s1;
          l2 = size2 - s2;
        }
        else{
          l1 = segments1.a[i+1] - s1;
          l2 = segments2.a[i+1] - s2;
        }
        // Set the max segment length for memory check
        if (i == low || m1 < l1) {
          m1 = l1;
        }
        if (i == low || m2 < l2) {
          m2 = l2;
        }
      }
      
      // perform memory exhaustion check using the size of the largest segment present
      var itemsize = if gEnt2.dtype == DType.UInt64 then numBytes(uint) else numBytes(int);
      var sortMem1 = radixSortLSD_memEst(m1, itemsize);
      var sortMem2 = radixSortLSD_memEst(m2, itemsize);
      var union_maxMem = max(sortMem1, sortMem2);
      overMemLimit(union_maxMem);

      select(gEnt2.dtype, gEnt4.dtype) {
        when (DType.Int64, DType.Int64) {
          var values1 = toSymEntry(gEnt2, int);
          var values2 = toSymEntry(gEnt4, int);

          var union_lens: [segments1.aD] int;

          // Compute lengths of the segments resulting from each union
          forall (i, s1, l1, s2, l2, ul) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, union_lens) with (var agg = newSrcAggregator(int)){
            // TODO - update to use lowLevelLocalizingSlice 
            var u = union1d(values1.a[s1..#l1], values2.a[s2..#l2]);
            agg.copy(ul, u.size);
          }

          const union_segs = (+ scan union_lens) - union_lens;
          var union_vals = makeDistArray((+ reduce union_lens), int);

          // Compute the union and add values to the corresponding indexes in values
          forall (i, s1, l1, s2, l2, us, ul) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, union_segs, union_lens) with (var agg = newSrcAggregator(int)){
            // TODO - update to use lowLevelLocalizingSlice 
            var u = union1d(values1.a[s1..#l1], values2.a[s2..#l2]);
            for i in (0..#ul){
              agg.copy(union_vals[i+us], u[i]);
            }
          }

          var s_name = st.nextName();
          st.addEntry(s_name, new shared SymEntry(union_segs));
          var v_name = st.nextName();
          st.addEntry(v_name, new shared SymEntry(union_vals));

          repMsg = "created " + st.attrib(s_name) + "+created " + st.attrib(v_name);
          asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
          return new MsgTuple(repMsg, MsgType.NORMAL);
        }
        when (DType.UInt64, DType.UInt64) {
          var values1 = toSymEntry(gEnt2, uint);
          var values2 = toSymEntry(gEnt4, uint);

          var union_lens: [segments1.aD] int;

          // Compute lengths of the segments resulting from each union
          forall (i, s1, l1, s2, l2, ul) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, union_lens) with (var agg = newSrcAggregator(int)){
            // TODO - update to use lowLevelLocalizingSlice 
            var u = union1d(values1.a[s1..#l1], values2.a[s2..#l2]);
            agg.copy(ul, u.size);
          }

          const union_segs = (+ scan union_lens) - union_lens;
          var union_vals = makeDistArray((+ reduce union_lens), uint);

          // Compute the union and add values to the corresponding indexes in values
          forall (i, s1, l1, s2, l2, us, ul) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2, union_segs, union_lens) with (var agg = newSrcAggregator(uint)){
            // TODO - update to use lowLevelLocalizingSlice 
            var u = union1d(values1.a[s1..#l1], values2.a[s2..#l2]);
            for i in (0..#ul){
              agg.copy(union_vals[i+us], u[i]);
            }
          }

          var s_name = st.nextName();
          st.addEntry(s_name, new shared SymEntry(union_segs));
          var v_name = st.nextName();
          st.addEntry(v_name, new shared SymEntry(union_vals));

          repMsg = "created " + st.attrib(s_name) + "+created " + st.attrib(v_name);
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
      registerFunction("union1d_multi", union1d_multiMsg, getModuleName());
    }
}
