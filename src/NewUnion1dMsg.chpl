module NewUnion1dMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    use Reflection only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedArray;
    use ServerErrorStrings;

    use NewUnion1d;

    proc newUnion1dMsg(reqMsg: string, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;
        var (cmd, name, name2) = reqMsg.splitMsgToTuple(3);

        var vname = st.nextName();

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        var gEnt2: borrowed GenSymEntry = st.lookup(name2);
        
        var e = toSymEntry(gEnt,int);
        var f = toSymEntry(gEnt2, int);
        
        var aV = newUnion1d(e.a, f.a);
        st.addEntry(vname, new shared SymEntry(aV));
                
        var s = try! "created " + st.attrib(vname);
        return s;
    }
}