
module In1dMsg
{
    use ServerConfig;

    use Reflection only;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use In1d;
    
    /*
    Small bound const. Brute force in1d implementation recommended.
    */
    private config const sBound = 2**4; 

    /*
    Medium bound const. Per locale associative domain in1d implementation recommended.
    */
    private config const mBound = 2**25; 

    /* in1d takes two pdarray and returns a bool pdarray
       with the "in"/contains for each element tested against the second pdarray.
       
       in1dMsg processes the request, considers the size of the arguements, and decides which implementation
       of in1d to utilize.
    */
    proc in1dMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, sname, flag) = payload.decode().splitMsgToTuple(3);
        var invert: bool;
        if flag == "True" {invert = true;}
        else if flag == "False" {invert = false;}
        else {return try! "Error: %s: %s".format(pn,flag);}

        // get next symbol name
        var rname = st.nextName();
        if v {try! writeln("%s %s %s %s : %s".format(cmd, name, sname, invert, rname));try! stdout.flush();}

        var gAr1: borrowed GenSymEntry = st.lookup(name);
        var gAr2: borrowed GenSymEntry = st.lookup(sname);

        select (gAr1.dtype, gAr2.dtype) {
            when (DType.Int64, DType.Int64) {
                var ar1 = toSymEntry(gAr1,int);
                var ar2 = toSymEntry(gAr2,int);

                // things to do...
                // if ar2 is big for some value of big... call unique on ar2 first

                // brute force if below small bound
                if (ar2.size <= sBound) {
                    if v {try! writeln("%t <= %t, using GlobalAr2Bcast".format(ar2.size,sBound)); try! stdout.flush();}

                    var truth = in1dGlobalAr2Bcast(ar1.a, ar2.a);
                    if (invert) {truth = !truth;}
                    
                    st.addEntry(rname, new shared SymEntry(truth));
                }
                // per locale assoc domain if below medium bound
                else if (ar2.size <= mBound) {
                    if v {try! writeln("%t <= %t, using Ar2PerLocAssoc".format(ar2.size,mBound)); try! stdout.flush();}
                    
                    var truth = in1dAr2PerLocAssoc(ar1.a, ar2.a);
                    if (invert) {truth = !truth;}
                    
                    st.addEntry(rname, new shared SymEntry(truth));
                }
                // sort-based strategy if above medium bound
                else {
                    if v {try! writeln("%t > %t, using sort-based strategy".format(ar2.size, mBound)); try! stdout.flush();}
                    var truth = in1dSort(ar1.a, ar2.a);
                    if (invert) {truth = !truth;}

                    st.addEntry(rname, new shared SymEntry(truth));
                }
                
            }
            otherwise {return notImplementedError(pn,gAr1.dtype,"in",gAr2.dtype);}
        }
        
        return try! "created " + st.attrib(rname);
    }

    
}
