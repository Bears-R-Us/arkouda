// arg sort algorithm
// these pass back an index vector which can be used
// to permute the original array into sorted order
//
//
module ArgSortMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use AryUtil;
    
    // thresholds for different sized sorts
    var lgSmall = 10;
    var small = 2**lgSmall;
    var lgMedium = 20;
    var medium = 2**lgMedium;
    var lgLarge = 30;
    var large = 2**lgLarge;

    // thresholds for ranges of values in the sorts
    var sBins = 2**10;
    var mBins = 2**25;
    var lBins = 2**25 * numLocales;

    proc +(x: atomic int, y: atomic int) {
        return x.read() + y.read();
    }
    
    proc +=(X: [?D] int, Y: [D] atomic int) {
        [i in D] {X[i] += Y[i].read();}
    }

    // do a counting sort on a (an array of integers)
    // returns iv an array of indices that would sort the array original array
    proc argCountSortGlobHist(a: [?aD] int, aMin: int, aMax: int): [aD] int {
        // index vector to hold permutation
        var iv: [aD] int;

        // how many bins in histogram
        var bins = aMax-aMin+1;
        if v {try! writeln("bins = %t".format(bins));}

        // perf improvement: what is a better strategy
        //     below a threshold on buckets
        //     use second dim on locales??? then + reduce across locales
        //     can we keep all atomics local??? I think so...
        //     var hD = {a_min..a_max} dmapped Replicated();// look at primer
        //     add up values from other locales into this locale's hist
        //     coforall loc in Locales { on loc {...;} }
        //     calc ends and starts per locale
        //     etc...
        // histogram domain size should be equal to a_nvals
        var hD = makeDistDom(bins);
        // atomic histogram
        var atomic_hist: [hD] atomic int;
        // normal histogram for + scan
        var hist: [hD] int;

        // count number of each value into atomic histogram
        [e in a] atomic_hist[e-aMin].add(1);
        // copy from atomic histogram to normal histogram
        [(e,ae) in zip(hist, atomic_hist)] e = ae.read();
        if v {printAry("hist =",hist);}

        // calc starts and ends of buckets
        var ends: [hD] int = + scan hist;
        if v {printAry("ends =",ends);}
        var starts: [hD] int = ends - hist;
        if v {printAry("starts =",starts);}

        // atomic position in output array for buckets
        var atomic_pos: [hD] atomic int;
        // copy in start positions
        [(ae,e) in zip(atomic_pos, starts)] ae.write(e);

        // permute index vector
        forall (e,i) in zip(a,aD) {
            var pos = atomic_pos[e-aMin].fetchAdd(1);// get position to deposit element
            iv[pos] = i;
        }
        
        // return the index vector
        return iv;
    }

    // do a counting sort on a (an array of integers)
    // returns iv an array of indices that would sort the array original array
    proc argCountSortLocHistGlobHist(a: [?aD] int, aMin: int, aMax: int): [aD] int {
        // index vector to hold permutation
        var iv: [aD] int;

        // how many bins in histogram
        var bins = aMax-aMin+1;
        if v {try! writeln("bins = %t".format(bins));}

        // perf improvement: what is a better strategy
        //     below a threshold on buckets
        //     use second dim on locales??? then + reduce across locales
        //     can we keep all atomics local??? I think so...
        //     var hD = {a_min..a_max} dmapped Replicated();// look at primer
        //     add up values from other locales into this locale's hist
        //     coforall loc in Locales { on loc {...;} }
        //     calc ends and starts per locale
        //     etc...

        // create a global count array to scan
        var globalCounts = makeDistArray(bins * numLocales, int);

        coforall loc in Locales {
            on loc {
                // histogram domain size should be equal to bins
                var hD = {0..#bins};

                // atomic histogram
                var atomicHist: [hD] atomic int;

                // count number of each value into local atomic histogram
                [i in a.localSubdomain()] atomicHist[a[i]-aMin].add(1);

                // put counts into globalCounts array
                [i in hD] globalCounts[i * numLocales + here.id] = atomicHist[i].read();
            }
        }

        // scan globalCounts to get bucket ends on each locale
        var globalEnds: [globalCounts.domain] int = + scan globalCounts;
        if v {printAry("globalCounts =",globalCounts);}
        if v {printAry("globalEnds =",globalEnds);}
        
        coforall loc in Locales {
            on loc {
                // histogram domain size should be equal to bins
                var hD = {0..#bins};
                var localCounts: [hD] int;
                [i in hD] localCounts[i] = globalCounts[i * numLocales + here.id];
                var localEnds: [hD] int = + scan localCounts;
                
                // atomic histogram
                var atomicHist: [hD] atomic int;

                // local storage to sort into
                var localBuffer: [0..#(a.localSubdomain().size)] int;

                // put locale-bucket-ends into atomic hist
                [i in hD] atomicHist[i].write(localEnds[i] - localCounts[i]);
                
                // get position in localBuffer of each element and place it there
                // counting up to local-bucket-end
                [idx in a.localSubdomain()] {
                    var pos = atomicHist[a[idx]-aMin].fetchAdd(1); // local pos in localBuffer
                    localBuffer[pos] = idx; // should be local pos and global idx
                }

                // move blocks to output array
                [i in hD] {
                    var gEnd = globalEnds[i * numLocales + here.id];
                    var gHigh = gEnd - 1;
                    var gLow =  gEnd - localCounts[i];
                    var lHigh = localEnds[i] - 1;
                    var lLow = localEnds[i] - localCounts[i];
                    if (gLow..gHigh).size != (lLow..lHigh).size {
                        writeln(gLow..gHigh, " ", lLow..lHigh);
                        writeln((gLow..gHigh).size, " != ", (lLow..lHigh).size);
                        try! stdout.flush();
                        exit(1);
                    }
                    if localCounts[i] > 0 {iv[gLow..gHigh] = localBuffer[lLow..lHigh];}
                }
            }
        }
        
        // return the index vector
        return iv;
    }

    // argsort takes pdarray and returns an index vector iv which sorts the array
    proc argsortMsg(reqMsg: string, st: borrowed SymTab): string {
        var pn = "argsort";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];

        // get next symbol name
        var ivname = st.nextName();
        if v {try! writeln("%s %s : %s %s".format(cmd, name, ivname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError(pn,name);}

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var eMin:int = min reduce e.a;
                var eMax:int = max reduce e.a;

                // how many bins/values possible in sort
                var bins = eMax-eMin+1;
                if v {try! writeln("bins = %t".format(bins));try! stdout.flush();}

                if (bins <= mBins) {
                    if v {try! writeln("%t <= %t".format(bins, mBins));try! stdout.flush();}
                    var iv = argCountSortLocHistGlobHist(e.a, eMin, eMax);
                    st.addEntry(ivname, new shared SymEntry(iv));
                }
                else {
                    if v {try! writeln("bins = %t".format(bins));try! stdout.flush();}
                    var iv = argCountSortGlobHist(e.a, eMin, eMax);
                    st.addEntry(ivname, new shared SymEntry(iv));
                }
            }
            otherwise {return notImplementedError(pn,gEnt.dtype);}
        }
        
        return try! "created " + st.attrib(ivname);
    }

}
