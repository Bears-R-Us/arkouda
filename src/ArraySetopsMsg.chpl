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
    proc intersect1dMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, name2, assume_unique) = payload.decode().splitMsgToTuple(3);
        var isUnique = stringtobool(assume_unique);
        
        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        var gEnt2: borrowed GenSymEntry = st.lookup(name2);
        
        var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
        var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
        var intersect_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
        overMemLimit(intersect_maxMem);

        select(gEnt.dtype) {
          when (DType.Int64) {
             if(gEnt.dtype != gEnt2.dtype) then return notImplementedError("newIntersect1d",gEnt2.dtype);
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2, int);
             
             var aV = intersect1d(e.a, f.a, isUnique);
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
    proc setxor1dMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, name2, assume_unique) = payload.decode().splitMsgToTuple(3);
        var isUnique = stringtobool(assume_unique);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        var gEnt2: borrowed GenSymEntry = st.lookup(name2);

        var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
        var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
        var xor_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
        overMemLimit(xor_maxMem);

        select(gEnt.dtype) {
          when (DType.Int64) {
             if(gEnt.dtype != gEnt2.dtype) then return notImplementedError("setxor1d",gEnt2.dtype);
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2, int);
             
             var aV = setxor1d(e.a, f.a, isUnique);
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
    proc setdiff1dMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, name2, assume_unique) = payload.decode().splitMsgToTuple(3);
        var isUnique = stringtobool(assume_unique);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        var gEnt2: borrowed GenSymEntry = st.lookup(name2);

        var gEnt_sortMem = radixSortLSD_memEst(gEnt.size, gEnt.itemsize);
        var gEnt2_sortMem = radixSortLSD_memEst(gEnt2.size, gEnt2.itemsize);
        var diff_maxMem = max(gEnt_sortMem, gEnt2_sortMem);
        overMemLimit(diff_maxMem);

        select(gEnt.dtype) {
          when (DType.Int64) {
             if(gEnt.dtype != gEnt2.dtype) then return notImplementedError("setdiff1d",gEnt2.dtype);
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2, int);
             
             var aV = setdiff1d(e.a, f.a, isUnique);
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
    proc union1dMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
      param pn = Reflection.getRoutineName();
      var repMsg: string; // response message
        // split request into fields
      var (name, name2) = payload.decode().splitMsgToTuple(2);

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

    proc stringtobool(str: string): bool throws {
      if str == "True" then return true;
      else if str == "False" then return false;
      throw new owned ErrorWithMsg("message: assume_unique must be of type bool");
    }
}
