/* unique finding and counting algorithms
 these are all based on dense histograms and sparse histograms(assoc domains/arrays)

 you could also use a sort if you got into a real bind with really
 large dense ranges of values and large arrays...

 *** need to factor in sparsity estimation somehow ***
 for example if (a.max-a.min > a.size) means that a's are sparse

 */
module UniqueMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    use Reflection only;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use Unique;
    
    /* unique take a pdarray and returns a pdarray with the unique values */
    proc uniqueMsg(reqMsg: string, st: borrowed SymTab): string {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        // flag to return counts of each unique value
        // same size as unique array
        var returnCounts: bool;
        if fields[3] == "True" {returnCounts = true;}
        else if fields[3] == "False" {returnCounts = false;}
        else {return try! "Error: %s: %s".format(pn,fields[3]);}
        
        // get next symbol name for unique
        var vname = st.nextName();
        // get next symbol anme for counts
        var cname = st.nextName();
        if v {try! writeln("%s %s %t: %s %s".format(cmd, name, returnCounts, vname, cname));try! stdout.flush();}
        
        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError(pn,name);}
        
        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                
                /* var eMin:int = min reduce e.a; */
                /* var eMax:int = max reduce e.a; */
                
                /* // how many bins in histogram */
                /* var bins = eMax-eMin+1; */
                /* if v {try! writeln("bins = %t".format(bins));try! stdout.flush();} */

                /* if (bins <= mBins) { */
                /*     if v {try! writeln("bins <= %t".format(mBins));try! stdout.flush();} */
                /*     var (aV,aC) = uniquePerLocHistGlobHist(e.a, eMin, eMax); */
                /*     st.addEntry(vname, new shared SymEntry(aV)); */
                /*     if returnCounts {st.addEntry(cname, new shared SymEntry(aC));} */
                /* } */
                /* else { */
                /*     if v {try! writeln("bins = %t".format(bins));try! stdout.flush();} */
                /*     var (aV,aC) = uniquePerLocAssocParUnsafeGlobAssocParUnsafe(e.a, eMin, eMax); */
                /*     st.addEntry(vname, new shared SymEntry(aV)); */
                /*     if returnCounts {st.addEntry(cname, new shared SymEntry(aC));} */
                /* } */

                var (aV,aC) = uniqueSort(e.a);
                st.addEntry(vname, new shared SymEntry(aV));
                if returnCounts {st.addEntry(cname, new shared SymEntry(aC));}
                    
            }
            otherwise {return notImplementedError("unique",gEnt.dtype);}
        }
        
        var s = try! "created " + st.attrib(vname);
        if returnCounts {s += " +created " + st.attrib(cname);}

        return s;
    }
    
    /* value_counts takes a pdarray and returns two pdarrays unique values and counts for each value */
    proc value_countsMsg(reqMsg: string, st: borrowed SymTab): string {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];

        // get next symbol name
        var vname = st.nextName();
        var cname = st.nextName();
        if v {try! writeln("%s %s : %s %s".format(cmd, name, vname, cname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError(pn,name);}

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                /* var eMin:int = min reduce e.a; */
                /* var eMax:int = max reduce e.a; */

                /* // how many bins in histogram */
                /* var bins = eMax-eMin+1; */
                /* if v {try! writeln("bins = %t".format(bins));try! stdout.flush();} */

                /* if (bins <= mBins) { */
                /*     if v {try! writeln("bins <= %t".format(mBins));try! stdout.flush();} */
                /*     var (aV,aC) = uniquePerLocHistGlobHist(e.a, eMin, eMax); */
                /*     st.addEntry(vname, new shared SymEntry(aV)); */
                /*     st.addEntry(cname, new shared SymEntry(aC)); */
                /* } */
                /* else if (bins <= lBins) { */
                /*     if v {try! writeln("bins <= %t".format(lBins));try! stdout.flush();} */
                /*     var (aV,aC) = uniquePerLocAssocGlobHist(e.a, eMin, eMax); */
                /*     st.addEntry(vname, new shared SymEntry(aV)); */
                /*     st.addEntry(cname, new shared SymEntry(aC)); */
                /* } */
                /* else { */
                /*     if v {try! writeln("bins = %t".format(bins));try! stdout.flush();} */
                /*     var (aV,aC) = uniquePerLocAssocGlobAssoc(e.a, eMin, eMax); */
                /*     st.addEntry(vname, new shared SymEntry(aV)); */
                /*     st.addEntry(cname, new shared SymEntry(aC)); */
                /* } */

                var (aV,aC) = uniqueSort(e.a);
                st.addEntry(vname, new shared SymEntry(aV));
                st.addEntry(cname, new shared SymEntry(aC));
            }
            otherwise {return notImplementedError(pn,gEnt.dtype);}
        }
        
        return try! "created " + st.attrib(vname) + " +created " + st.attrib(cname);
    }

}

