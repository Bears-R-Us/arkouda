// unique finding and counting algorithms
// these are all based on dense histograms and sparse histograms(assoc domains/arrays)
//
// you could also use a sort if you got into a real bind with really
// large dense ranges of values and large arrays...
//
// *** need to factor in sparsity estimation somehow ***
// for example if (a.max-a.min > a.size) means that a's are sparse
//
//
module UniqueMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use PrivateDist;
    //use HashedDist;

    // thresholds for different unique counting algorithms
    var sBins = 2**10; // small-range maybe for using reduce intents on forall loops
    var mBins = 2**20; // medium-range per-locale atomic hist
    var lBins = 2**26 * numLocales; // large-range per-locale assoc array + global atomic hist
    
    proc +(x: atomic int, y: atomic int) {
        return x.read() + y.read();
    }
    
    proc +=(X: [?D] int, Y: [D] atomic int) {
        forall i in D do 
            X[i] += Y[i].read();
    }

    // unique with global histogram
    proc uniqueGlobHist(a: [?aD] int, aMin: int, aMax: int) {
        // how many bins in histogram
        var bins = aMax-aMin+1;

        // distributed histogram domain
        var hD = makeDistDom(bins);

        // allocate atomic histogram
        var atomicHist: [hD] atomic int;

        // count into atomic histogram
        forall val in a {
            var bin = val - aMin;
            if val == aMax {bin = bins-1;}
            atomicHist[bin].add(1);
        }

        // integer truth/non-empty-bin array
        var itruth = makeDistArray(bins,int);

        // find non-empty bins in atomic histogram
        [(t,ae) in zip(itruth, atomicHist)] t = (ae.read() != 0):int;

        // calc indices of the non-zero count elements
        var iv: [hD] int = (+ scan itruth);

        // how many entries in unique array
        var pop = iv[iv.size-1];

        // unique array
        var aV = makeDistArray(pop, int);
        // counts array
        var aC = makeDistArray(pop, int);

        // if value has non-zero histogram bin
        // put it into the unique array
        // and its count into the count array
        [i in hD] if (itruth[i] == 1) {
            aV[iv[i]-1] = i+aMin;
            aC[iv[i]-1] = atomicHist[i].read();
        }// iv[i]-1 for zero base index
        
        return (aV, aC);
    }

    // unique with per-locale histograms
    proc uniquePerLocHistGlobHist(a: [?aD] int, aMin: int, aMax: int) {

        // how many bin in histogram
        var bins = aMax-aMin+1;

        // allocate per-locale atomic histogram
        var atomicHist: [PrivateSpace] [0..#bins] atomic int;

        // count into per-locale private atomic histogram
        forall val in a {
            var bin = val - aMin;
            if val == aMax {bin = bins-1;}
            atomicHist[here.id][bin].add(1);
        }

        // distributed histogram domain
        var hD = makeDistDom(bins);
        
        // +reduce across per-locale histograms to get counts
        var globalHist: [hD] int = + reduce [i in PrivateSpace] atomicHist[i];

        // integer truth/non-empty-bin array
        var itruth: [hD] int;
        
        // find non-empty bins
        [(t,e) in zip(itruth, globalHist)] t = (e != 0):int;

        // calc indices of the non-zero count elements
        var iv: [hD] int = (+ scan itruth);

        // how many entries in unique array
        var pop = iv[iv.size-1];

        // unique array
        var aV = makeDistArray(pop, int);
        // counts array
        var aC = makeDistArray(pop, int);

        // if value has non-zero histogram bin
        // put value into the unique array
        // and value's count into the count array
        [i in hD] if (itruth[i] == 1) {
            aV[iv[i]-1] = i+aMin;
            aC[iv[i]-1] = globalHist[i];
        }// iv[i]-1 for zero base index
        
        return (aV, aC);
    }

    // use when unique value vary over a wide range and and are sparse
    // unique with per-locale assoc domains and arrays
    // global unique value histogram
    proc uniquePerLocAssocGlobHist(a: [?aD] int, aMin: int, aMax: int) {

        // per locale assoc domain of int to hold uniq values
        var uniqSet: [PrivateSpace] domain(int);

        // accumulate the uniq values into each locales domain of uniq values
        [val in a] if !uniqSet[here.id].contains(val) {uniqSet[here.id] += val;}

        // how many bins in histogram
        var bins = aMax-aMin+1;

        // distributed histogram domain
        var hD = makeDistDom(bins);

        // allocate atomic histogram
        var atomicHist: [hD] atomic int;
        
        // calc local counts and then effectively +reduce uniqCounts to get global histogram
        coforall loc in Locales {
            on loc {
                var uniqCounts: [uniqSet[here.id]] atomic int;

                // count local part of array's values into per-locale private atomic counter set
                [i in a.localSubdomain()] uniqCounts[a[i]].add(1);

                // add local counts for unique value to global histogram
                [val in uniqSet[here.id]] atomicHist[val - aMin].add(uniqCounts[val].read());
            }
        }

        // integer truth/non-empty-bin array
        var itruth = makeDistArray(bins,int);

        // find non-empty bins in atomic histogram
        [(t,ae) in zip(itruth, atomicHist)] t = (ae.read() != 0):int;

        // calc indices of the non-zero count elements
        var iv: [hD] int = (+ scan itruth);

        // how many entries in unique array
        var pop = iv[iv.size-1];

        // unique array
        var aV = makeDistArray(pop, int);
        // counts array
        var aC = makeDistArray(pop, int);

        // if value has non-zero histogram bin
        // put it into the unique array
        // and its count into the count array
        [i in hD] if (itruth[i] == 1) {
            aV[iv[i]-1] = i+aMin;
            aC[iv[i]-1] = atomicHist[i].read();
        }// iv[i]-1 for zero base index

        return (aV, aC);
    }

    // use when unique value vary over a wide range and and are sparse
    // unique with per-locale assoc domains and arrays
    proc uniquePerLocAssocGlobAssoc(a: [?aD] int, aMin: int, aMax: int) {

        // per locale assoc domain of int to hold uniq values
        var uniqSet: [PrivateSpace] domain(int);

        // accumulate the uniq values into each locales domain of uniq values
        [val in a] if !uniqSet[here.id].contains(val) {uniqSet[here.id] += val;}
        var numUniq = + reduce [i in PrivateSpace]  uniqSet[i].size;
        if v {try! writeln("num unique vals upper bound = %t".format(numUniq));try! stdout.flush();}

        // global assoc domain for global unique value set
        //var globalUniqSet: domain(int) dmapped Hashed(idxType=int);
        var globalUniqSet: domain(int);
        
        // efectively +reduce(union-reduction) private uniqSet domians to get global uniqSet
        // what I really want is:
        //[i in PrivateSpace] globalUniqSet += uniqSet[i];
        // or maybe even...
        for i in PrivateSpace {globalUniqSet += uniqSet[i];}
        // ok, well this... only one that works for HashedDist Assoc Domain
        //for loc in Locales { on loc {
        //         for val in uniqSet[here.id] {globalUniqSet += val;}
        //     }
        // }
        if v {try! writeln("num unique vals = %t".format(globalUniqSet.size));try! stdout.flush();}

        // allocate global uniqCounts over global set of uniq values
        var globalUniqCounts: [globalUniqSet] atomic int;
        
        // calc local counts and then effectively +reduce uniqCounts to get global uniqCount
        coforall loc in Locales {
            on loc {
                var uniqCounts: [uniqSet[here.id]] atomic int;

                // count locale part of array's values into per-locale private atomic counter set
                [i in a.localSubdomain()] uniqCounts[a[i]].add(1);

                // accumulate into global counters
                [val in uniqSet[here.id]] globalUniqCounts[val].add(uniqCounts[val].read());
            }
        }

        // unique array
        var aV = makeDistArray(globalUniqSet.size, int);
        // counts array
        var aC = makeDistArray(globalUniqSet.size, int);

        var idx: atomic int;
        [val in globalUniqSet] {
            var i = idx.fetchAdd(1); // get index into dist array
            aV[i] = val; // copy unique value
            aC[i] = globalUniqCounts[val].read(); // copy count of unique value
        }
        
        // unlike the other versions
        // these are prob not sorted by value
        return (aV, aC);
    }
    
    // unique take a pdarray and returns a pdarray with the unique values
    proc uniqueMsg(reqMsg: string, st: borrowed SymTab): string {
        var pn = "unique";
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
                var eMin:int = min reduce e.a;
                var eMax:int = max reduce e.a;
                
                // how many bins in histogram
                var bins = eMax-eMin+1;
                if v {try! writeln("bins = %t".format(bins));try! stdout.flush();}

                if (bins <= mBins) {
                    if v {try! writeln("bins <= %t".format(mBins));try! stdout.flush();}
                    var (aV,aC) = uniquePerLocHistGlobHist(e.a, eMin, eMax);
                    st.addEntry(vname, new shared SymEntry(aV));
                    if returnCounts {st.addEntry(cname, new shared SymEntry(aC));}
                }
                else if (bins <= lBins) {
                    if v {try! writeln("bins <= %t".format(lBins));try! stdout.flush();}
                    var (aV,aC) = uniquePerLocAssocGlobHist(e.a, eMin, eMax);
                    st.addEntry(vname, new shared SymEntry(aV));
                    if returnCounts {st.addEntry(cname, new shared SymEntry(aC));}
                }
                else {
                    if v {try! writeln("bins = %t".format(bins));try! stdout.flush();}
                    var (aV,aC) = uniquePerLocAssocGlobAssoc(e.a, eMin, eMax);
                    st.addEntry(vname, new shared SymEntry(aV));
                    if returnCounts {st.addEntry(cname, new shared SymEntry(aC));}
                }
                    
            }
            otherwise {return notImplementedError("unique",gEnt.dtype);}
        }
        
        var s = try! "created " + st.attrib(vname);
        if returnCounts {s += " +created " + st.attrib(cname);}

        return s;
    }
    
    // value_counts rtakes a pdarray and returns two pdarrays unique values and counts for each value
    proc value_countsMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];

        // get next symbol name
        var vname = st.nextName();
        var cname = st.nextName();
        if v {try! writeln("%s %s : %s %s".format(cmd, name, vname, cname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError("value_counts",name);}

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var eMin:int = min reduce e.a;
                var eMax:int = max reduce e.a;

                // how many bins in histogram
                var bins = eMax-eMin+1;
                if v {try! writeln("bins = %t".format(bins));try! stdout.flush();}

                if (bins <= mBins) {
                    if v {try! writeln("bins <= %t".format(mBins));try! stdout.flush();}
                    var (aV,aC) = uniquePerLocHistGlobHist(e.a, eMin, eMax);
                    st.addEntry(vname, new shared SymEntry(aV));
                    st.addEntry(cname, new shared SymEntry(aC));
                }
                else if (bins <= lBins) {
                    if v {try! writeln("bins <= %t".format(lBins));try! stdout.flush();}
                    var (aV,aC) = uniquePerLocAssocGlobHist(e.a, eMin, eMax);
                    st.addEntry(vname, new shared SymEntry(aV));
                    st.addEntry(cname, new shared SymEntry(aC));
                }
                else {
                    if v {try! writeln("bins = %t".format(bins));try! stdout.flush();}
                    var (aV,aC) = uniquePerLocAssocGlobAssoc(e.a, eMin, eMax);
                    st.addEntry(vname, new shared SymEntry(aV));
                    st.addEntry(cname, new shared SymEntry(aC));
                }
            }
            otherwise {return notImplementedError("value_counts",gEnt.dtype);}
        }
        
        return try! "created " + st.attrib(vname) + " +created " + st.attrib(cname);
    }

}

