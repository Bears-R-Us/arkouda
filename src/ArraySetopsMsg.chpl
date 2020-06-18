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
    
    /*
    Parse, execute, and respond to a intersect1d message
    :arg reqMsg: request containing (cmd,name,name2,assume_unique)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (string) response message
    */
    proc intersect1dMsg(reqMsg: string, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (cmd, name, name2, assume_unique) = reqMsg.splitMsgToTuple(4);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        var gEnt2: borrowed GenSymEntry = st.lookup(name2);
        overMemLimit(((4 + 1) * 2 * gEnt.size * gEnt.itemsize)
             + (2 * here.maxTaskPar * numLocales * 2**16 * 8));

        select(gEnt.dtype) {
          when (DType.Int64) {
             if(gEnt.dtype != gEnt2.dtype) then return notImplementedError("newIntersect1d",gEnt2.dtype);
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2, int);
             
             var aV = intersect1d(e.a, f.a, assume_unique);
             st.addEntry(vname, new shared SymEntry(aV));

             var s = try! "created " + st.attrib(vname);
             return s;
           }
           otherwise {
             return notImplementedError("newIntersect1d",gEnt.dtype);
           }
        }
    }

    /*
    Parse, execute, and respond to a setxor1d message
    :arg reqMsg: request containing (cmd,name,name2,assume_unique)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (string) response message
    */
    proc setxor1dMsg(reqMsg: string, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (cmd, name, name2, assume_unique) = reqMsg.splitMsgToTuple(4);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        var gEnt2: borrowed GenSymEntry = st.lookup(name2);
        overMemLimit(((4 + 1) * 2 * gEnt.size * gEnt.itemsize)
             + (2 * here.maxTaskPar * numLocales * 2**16 * 8));

        select(gEnt.dtype) {
          when (DType.Int64) {
             if(gEnt.dtype != gEnt2.dtype) then return notImplementedError("setxor1d",gEnt2.dtype);
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2, int);
             
             var aV = setxor1d(e.a, f.a, assume_unique);
             st.addEntry(vname, new shared SymEntry(aV));

             var s = try! "created " + st.attrib(vname);
             return s;
           }
           otherwise {
             return notImplementedError("setxor1d",gEnt.dtype);
           }
        }
    }

    /*
    Parse, execute, and respond to a setdiff1d message
    :arg reqMsg: request containing (cmd,name,name2,assume_unique)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (string) response message
    */
    proc setdiff1dMsg(reqMsg: string, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (cmd, name, name2, assume_unique) = reqMsg.splitMsgToTuple(4);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        var gEnt2: borrowed GenSymEntry = st.lookup(name2);
        overMemLimit(((4 + 1) * 2 * gEnt.size * gEnt.itemsize)
             + (2 * here.maxTaskPar * numLocales * 2**16 * 8));

        select(gEnt.dtype) {
          when (DType.Int64) {
             if(gEnt.dtype != gEnt2.dtype) then return notImplementedError("setdiff1d",gEnt2.dtype);
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2, int);
             
             var aV = setdiff1d(e.a, f.a, assume_unique);
             st.addEntry(vname, new shared SymEntry(aV));

             var s = try! "created " + st.attrib(vname);
             return s;
           }
           otherwise {
             return notImplementedError("setdiff1d",gEnt.dtype);
           }
        }
    }

    /*
    Parse, execute, and respond to a union1d message
    :arg reqMsg: request containing (cmd,name,name2,assume_unique)
    :type reqMsg: string
    :arg st: SymTab to act on
    :type st: borrowed SymTab
    :returns: (string) response message
    */
    proc union1dMsg(reqMsg: string, st: borrowed SymTab): string throws {
      param pn = Reflection.getRoutineName();
      var repMsg: string; // response message
        // split request into fields
      var (cmd, name, name2) = reqMsg.splitMsgToTuple(3);

      var vname = st.nextName();

      var gEnt: borrowed GenSymEntry = st.lookup(name);
      var gEnt2: borrowed GenSymEntry = st.lookup(name2);
      var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
      var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
      var union_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
      overMemLimit(union_maxMem);

      select(gEnt.dtype) {
        when (DType.Int64) {
           if(gEnt.dtype != gEnt2.dtype) then return notImplementedError("newUnion1d",gEnt2.dtype);
           var e = toSymEntry(gEnt,int);
           var f = toSymEntry(gEnt2, int);

           var aV = union1d(e.a, f.a);
           st.addEntry(vname, new shared SymEntry(aV));

           var s = try! "created " + st.attrib(vname);
           return s;
         }
         otherwise {
           return notImplementedError("newUnion1d",gEnt.dtype);
         }
      }
    }
}