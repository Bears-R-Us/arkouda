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
    
    use PrivateDist;

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
        
        // check counts against ends
        if v {[(ae,e) in zip(atomic_pos,ends)]
                if ae.read() != e {writeln("mismatch in atomic counts!!!");}}
        
        // return the index vector
        return iv;
    }

    /* // do a counting sort on a (an array of integers) */
    /* // returns iv an array of indices that would sort the array original array */
    /* proc argCountSortLocHistGlobHist(a: [?D] int, aMin: int, aMax: int): [D] int { */
    /*     // index vector to hold permutation */
    /*     var iv: [D] int; */

    /*     // how many bins in histogram */
    /*     var a_nvals = aMax-aMin+1; */
    /*     if v {try! writeln("a_nvals = %t".format(a_nvals));} */

    /*     // perf improvement: what is a better strategy */
    /*     //     below a threshold on buckets */
    /*     //     use second dim on locales??? then + reduce across locales */
    /*     //     can we keep all atomics local??? I think so... */
    /*     //     var hD = {a_min..a_max} dmapped Replicated();// look at primer */
    /*     //     add up values from other locales into this locale's hist */
    /*     //     coforall loc in Locales { on loc {...;} } */
    /*     //     calc ends and starts per locale */
    /*     //     etc... */

    /*     // histogram domain size should be equal to a_nvals */
    /*     var hD: domain(1) = {aMin..aMax}; */

    /*     // atomic histogram */
    /*     var atomicHist: [PrivateSpace] [hD] atomic int; */

    /*     // count number of each value into atomic histogram */
    /*     [e in a] atomicHist[here.id][e].add(1); */
        
    /*     // global histogram for value-bucket sizes */
    /*     var globHist: [hD] int = + reduce [i in PrivateSpace] atomicHist[i]; */

    /*     // calc global starts and ends of value-buckets */
    /*     var globEnds: [hD] int = + scan globHist; */
    /*     if v {printAry("globEnds =",globEnds);} */
    /*     var globStarts: [hD] int = globEnds - globHist; */
    /*     if v {printAry("globStarts =", globStarts);} */

    /*     // calc per locale value-bucket ends and starts */
    /*     // + scan across locales */
        
        
    /*     //....... */
        
    /*     // atomic position in output array for buckets */
    /*     var atomic_pos: [hD] atomic int; */
    /*     // copy in start positions */
    /*     [(ae,e) in zip(atomic_pos, starts)] ae.write(e); */

    /*     // permute index vector */
    /*     for (e,i) in zip(a,aD) { */
    /*         var pos = atomic_pos[e].fetchAdd(1);// get position to deposit element */
    /*         iv[pos] = i; */
    /*     } */

    /*     // check counts against ends */
    /*     if v {[(ae,e) in zip(atomic_pos,ends)] */
    /*             if ae.read() != e {writeln("mismatch in atomic counts!!!");}} */
        
    /*     // return the index vector */
    /*     return iv; */
    /* } */

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
                    var iv = argCountSortGlobHist(e.a, eMin, eMax);
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
