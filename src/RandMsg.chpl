module RandMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    use Reflection only;
    use Random; // include everything from Random
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    /*
    parse, execute, and respond to randint message
    uniform int in half-open interval [min,max)

    :arg reqMsg: message to process (contains cmd,aMin,aMax,len,dtype)
    */
    proc randintMsg(reqMsg: string, st: borrowed SymTab): string {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var aMin = try! fields[2]:int;
        var aMax = try! fields[3]:int;
        var len = try! fields[4]:int;
        var dtype = str2dtype(fields[5]);

        // get next symbol name
        var rname = st.nextName();
        
        // if verbose print action
        if v {try! writeln("%s %i %i %i %s : %s".format(cmd,aMin,aMax,len,dtype2str(dtype),rname)); try! stdout.flush();}
        select (dtype) {
            when (DType.Int64) {                
                var t1 = Time.getCurrentTime();
                var aD = makeDistDom(len);
                var a = makeDistArray(len, int);
                writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                t1 = Time.getCurrentTime();
                coforall loc in Locales {
                    on loc {
		      ref myA = a.localSlice[a.localSubdomain()];
		      fillRandom(myA);
		      [ai in myA] if (ai < 0) { ai = -ai; }
		      const modulus = aMax - aMin;
		      myA = (myA % modulus) + aMin;
                    }
                }
                writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                st.addEntry(rname, new shared SymEntry(a));
            }
            when (DType.Float64) {
                var t1 = Time.getCurrentTime();
                var aD = makeDistDom(len);
                var a = makeDistArray(len, real);
                writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                t1 = Time.getCurrentTime();
                coforall loc in Locales {
                    on loc {
		      ref myA = a.localSlice[a.localSubdomain()];
		      fillRandom(myA);
		      const scale = aMax - aMin;
		      myA = scale*myA + aMin;
                    }
                }
                writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                st.addEntry(rname, new shared SymEntry(a));                
            }
            when (DType.Bool) {
                var t1 = Time.getCurrentTime();
                var aD = makeDistDom(len);
                var a = makeDistArray(len, bool);
                writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                t1 = Time.getCurrentTime();
                coforall loc in Locales {
                    on loc {
		      ref myA = a.localSlice[a.localSubdomain()];
		      fillRandom(myA);
                        /* var R = new owned RandomStream(real); R.getNext(); */
                        /* [i in a.localSubdomain()] a[i] = (R.getNext() >= 0.5); */
                    }
                }
                        writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
                
                st.addEntry(rname, new shared SymEntry(a));
            }            
            otherwise {return notImplementedError(pn,dtype);}
        }
        // response message
        return try! "created " + st.attrib(rname);
    }

}
