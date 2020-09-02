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
    use SegmentedArray;
    use ServerErrorStrings;

    use Unique;
    
    /* unique take a pdarray and returns a pdarray with the unique values */
    proc uniqueMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (objtype, name, returnCountsStr) = payload.decode().splitMsgToTuple(3);
        // flag to return counts of each unique value
        // same size as unique array
        var returnCounts: bool;
        if returnCountsStr == "True" {returnCounts = true;}
        else if returnCountsStr == "False" {returnCounts = false;}
        else {return try! "Error: %s: %s".format(pn,returnCountsStr);}
        select objtype {
          when "pdarray" {
            // get next symbol name for unique
            var vname = st.nextName();
            // get next symbol anme for counts
            var cname = st.nextName();
            if v {try! writeln("%s %s %t: %s %s".format(cmd, name, returnCounts, vname, cname));try! stdout.flush();}
        
            var gEnt: borrowed GenSymEntry = st.lookup(name);
            // the upper limit here is the same as argsort/radixSortLSD_keys
            // check and throw if over memory limit
            overMemLimit(((4 + 1) * gEnt.size * gEnt.itemsize)
                         + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
        
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
          when "str" {
            var offsetName = st.nextName();
            var valueName = st.nextName();
            var (names1,names2) = name.splitMsgToTuple('+', 2);
            var str = new owned SegString(names1, names2, st);
            // the upper limit here is the similar to argsort/radixSortLSD_keys, but with a few more scratch arrays
            // check and throw if over memory limit
            overMemLimit((8 * str.size * 8)
                         + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
            var (uo, uv, c, inv) = uniqueGroup(str);
            st.addEntry(offsetName, new shared SymEntry(uo));
            st.addEntry(valueName, new shared SymEntry(uv));
            var s = try! "created " + st.attrib(offsetName) + " +created " + st.attrib(valueName);
            if returnCounts {
              var countName = st.nextName();
              st.addEntry(countName, new shared SymEntry(c));
              s += " +created " + st.attrib(countName);
            }
            return s;
          }
          otherwise { return notImplementedError(Reflection.getRoutineName(), objtype); }
        }
    }
    
    /* value_counts takes a pdarray and returns two pdarrays unique values and counts for each value */
    proc value_countsMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name) = payload.decode().splitMsgToTuple(1);

        // get next symbol name
        var vname = st.nextName();
        var cname = st.nextName();
        if v {try! writeln("%s %s : %s %s".format(cmd, name, vname, cname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);

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

