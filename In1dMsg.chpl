
module In1dMsg
{
    use ServerConfig;

    use Time only;
    use Math only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use PrivateDist;

    var sBound = 2**6; // small bound do brute force
    var mBound = 2**25; // medium bound do per locale assoc domain

    /* brute force
    forward-way reduction per element of ar1 over ar2
    causes every elt in ar1 to be broadcast/communicated over ar2 */
    proc in1dGlobalAr1Bcast(ar1: [?aD1] int, ar2: [?aD2] int) {

        var truth = makeDistArray(ar1.size, bool);
        
        [(elt,t) in zip(ar1,truth)] t = | reduce (elt == ar2);

        return truth;
    }

    /* brute force
    reverse-way serial-or-reduce for each elt in ar2 over ar1
    causes every elt in ar2 to be broadcast/communicated over ar1 */
    proc in1dGlobalAr2Bcast(ar1: [?aD1] int, ar2: [?aD2] int) {

        var truth = makeDistArray(ar1.size, bool);

        for elt in ar2 {truth |= (ar1 == elt);}
        
        return truth;
    }

    /* put ar2 into an assoc domain of int per locale */
    proc in1dAr2PerLocAssoc(ar1: [?aD1] int, ar2: [?aD2] int) {

        var truth = makeDistArray(ar1.size, bool);
        var timings: [PrivateSpace] [0..#3] real;
        
        coforall loc in Locales {
            on loc {

                var t = new Time.Timer();
                
                if v {t.start();}
                var ar2Set: domain(int); // create a set to hold ar2
                ar2Set.requestCapacity(100_000); // requrest a capacity for the initial set
                if v {t.stop(); timings[here.id][0] = t.elapsed(); t.clear(); t.start();}

                ar2Set += ar2; // bulk add all the elts from ar2 to the set
                if v {t.stop(); timings[here.id][1] = t.elapsed(); t.clear(); t.start();}
                
                [i in truth.localSubdomain()] truth[i] = ar2Set.contains(ar1[i]);
                if v {t.stop(); timings[here.id][2] = t.elapsed();}
            }
        }

        if v {
            writeln("max create time = ",     max reduce [i in PrivateSpace] timings[i][0]);
            writeln("max fill time = ",       max reduce [i in PrivateSpace] timings[i][1]);
            writeln("max membership time = ", max reduce [i in PrivateSpace] timings[i][2]);
            try! stdout.flush();
        }
        
        return truth;
    }
    
    /* in1d takes two pdarray and returns a bool pdarray
       with the "in"/contains for each element tested against the second pdarray */
    proc in1dMsg(reqMsg: string, st: borrowed SymTab): string {
        var pn = "in1d";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var sname = fields[3];

        // get next symbol name
        var rname = st.nextName();
        if v {try! writeln("%s %s %s : %s".format(cmd, name, sname, rname));try! stdout.flush();}

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
                    
                    st.addEntry(rname, new shared SymEntry(truth));
                }
                // per locale assoc domain if below medium bound
                else if (ar2.size <= mBound) {
                    if v {try! writeln("%t <= %t".format(ar2.size,mBound)); try! stdout.flush();}
                    
                    var truth = in1dAr2PerLocAssoc(ar1.a, ar2.a);
                    
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
