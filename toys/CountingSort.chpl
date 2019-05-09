module CountingSort
{
    use AryUtil;
    
    // module level verbose flag
    var v = true;
    
    // do a counting sort on a (an array of integers)
    // returns iv an array of indices that would sort the array original array
    proc argCountSort(a: [?D] int): [D] int {
        // index vector to hold permutation
        var iv: [D] int;

        // stats on input array
        var (a_min,a_max,a_mean,a_vari,a_std) = aStats(a);
        if v {try! writeln("a_min = %t a_max = %t a_mean = %t \na_vari = %t a_std = %t".format(
                               a_min,a_max,a_mean,a_vari,a_std));}

        // how many bins in histogram
        var a_nvals = a_max-a_min+1;
        if v {try! writeln("a_nvals = %t".format(a_nvals));}

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
        var hD: domain(1) = {a_min..a_max};
        // atomic histogram
        var atomic_hist: [hD] atomic int;
        // normal histogram for + scan
        var hist: [hD] int;

        // count number of each value into atomic histogram
        [e in a] atomic_hist[e].add(1);
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
        for (e,i) in zip(a,a.domain) {
            var pos = atomic_pos[e].fetchAdd(1);// get position to deposit element
            iv[pos] = i;
        }

        // check counts against ends
        if v {[(ae,e) in zip(atomic_pos,ends)]
                if ae.read() != e {writeln("mismatch in atomic counts!!!");}}
        
        // return the index vector
        return iv;
    }

}
