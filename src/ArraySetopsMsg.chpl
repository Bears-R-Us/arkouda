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

    proc intersect1dMsg(reqMsg: string, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;
        var (cmd, name, name2, assume_unique) = reqMsg.splitMsgToTuple(4);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        var gEnt2: borrowed GenSymEntry = st.lookup(name2);

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

    proc setxor1dMsg(reqMsg: string, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;
        var (cmd, name, name2, assume_unique) = reqMsg.splitMsgToTuple(4);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        var gEnt2: borrowed GenSymEntry = st.lookup(name2);

        select(gEnt.dtype) {
          when (DType.Int64) {
             if(gEnt.dtype != gEnt2.dtype) then return notImplementedError("newUnion1d",gEnt2.dtype);
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2, int);
             
             var aV = setxor1d(e.a, f.a, assume_unique);
             st.addEntry(vname, new shared SymEntry(aV));

             var s = try! "created " + st.attrib(vname);
             return s;
           }
           otherwise {
             return notImplementedError("newUnion1d",gEnt.dtype);
           }
        }
    }

    proc setdiff1dMsg(reqMsg: string, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;
        var (cmd, name, name2, assume_unique) = reqMsg.splitMsgToTuple(4);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        var gEnt2: borrowed GenSymEntry = st.lookup(name2);

        select(gEnt.dtype) {
          when (DType.Int64) {
             if(gEnt.dtype != gEnt2.dtype) then return notImplementedError("newUnion1d",gEnt2.dtype);
             var e = toSymEntry(gEnt,int);
             var f = toSymEntry(gEnt2, int);
             
             var aV = setdiff1d(e.a, f.a, assume_unique);
             st.addEntry(vname, new shared SymEntry(aV));

             var s = try! "created " + st.attrib(vname);
             return s;
           }
           otherwise {
             return notImplementedError("newUnion1d",gEnt.dtype);
           }
        }
    }

    proc union1dMsg(reqMsg: string, st: borrowed SymTab): string throws {
      param pn = Reflection.getRoutineName();
      var repMsg: string;
      var (cmd, name, name2) = reqMsg.splitMsgToTuple(3);

      var vname = st.nextName();

      var gEnt: borrowed GenSymEntry = st.lookup(name);
      var gEnt2: borrowed GenSymEntry = st.lookup(name2);

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