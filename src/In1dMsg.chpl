
module In1dMsg
{
    use ServerConfig;

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
    proc in1dMsg(reqMsg: string, st: borrowed SymTab): string {
        var pn = "in1d";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var sname = fields[3];
        var invert: bool;
        if fields[4] == "True" {invert = true;}
        else if fields[4] == "False" {invert = false;}
        else {return try! "Error: %s: %s".format(pn,fields[4]);}

        // get next symbol name
        var rname = st.nextName();
        if v {try! writeln("%s %s %s %s : %s".format(cmd, name, sname, invert, rname));try! stdout.flush();}

        var gAr1: borrowed GenSymEntry = st.lookup(name);
        if (gAr1 == nil) {return unknownSymbolError("in1d",name);}
        var gAr2: borrowed GenSymEntry = st.lookup(sname);
        if (gAr2 == nil) {return unknownSymbolError("in1d",sname);}

        select (gAr1.dtype, gAr2.dtype) {
            when (DType.Int64, DType.Int64) {
                var ar1 = toSymEntry(gAr1,int);
                var ar2 = toSymEntry(gAr2,int);

                // things to do...
                // if ar2 is big for some value of big... call unique on ar2 first

                // brute force if below small bound
                if (ar2.size <= sBound) {
                    if v {try! writeln("%t <= %t".format(ar2.size,sBound)); try! stdout.flush();}

                    var truth = in1dGlobalAr2Bcast(ar1.a, ar2.a);
                    if (invert) {truth = !truth;}
                    
                    st.addEntry(rname, new shared SymEntry(truth));
                }
                // per locale assoc domain if below medium bound
                else if (ar2.size <= mBound) {
                    if v {try! writeln("%t <= %t".format(ar2.size,mBound)); try! stdout.flush();}
                    
                    var truth = in1dAr2PerLocAssoc(ar1.a, ar2.a);
                    if (invert) {truth = !truth;}
                    
                    st.addEntry(rname, new shared SymEntry(truth));
                }
                // error if above medium bound
                else {return try! "Error: %s: ar2 size too large %t".format(pn,ar2.size);}
                
            }
            otherwise {return notImplementedError(pn,gAr1.dtype,"in",gAr2.dtype);}
        }
        
        return try! "created " + st.attrib(rname);
    }

    
}
