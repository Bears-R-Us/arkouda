module RadixSort
{
    use AryUtil;

    use PrivateDist;
    
    // module level verbose flag
    var v = true;

    // thresholds and limits
    var lgRadix = 20; // lg2(radix)
    var radix = 1<<lgRadix; // power of two radix
    var radixMask = radix-1; // power of two minus 1 == ones mask for radix
    
    var sBound = radix;
    var mBound = 2**20;
    var lBound = 2**26 * numLocales;

    proc +(x: atomic int, y: atomic int) {
        return x.read() + y.read();
    }
    
    proc +=(X: [?D] int, Y: [D] atomic int) {
        [i in D] {X[i] += Y[i].read();}
    }
    
    // a is a tuple array of (v, origin-index)
    proc argRadixOnePass(a: [?aD] int, lgRadix: int, rShift: int): [aD] int {

        var radix = 1<<lgRadix;
        
        // allocate per-locale atomic histogram
        var atomicHist: [PrivateSpace] [0..#radix] atomic int;

        // count into per-locale private atomic histogram
        forall val in a {
            var bin = (val(0) >> rShift) & radixMask;
            atomicHist[here.id][bin].add(1);
        }

//...        
    }

    
    
    // radix sort on integers
    // returning iv an array of indices that would sort the array original array
    proc argRadixSort(a: [?D] int): [D] int {

        // index vector to hold permutation
        var iv: [D] int;

        // min and max values in a
        var aMin = min reduce a;
        var aMax = max reduce a;

        var bins = aMax-aMin+1;
        if v {try! writeln("bins = %t".format(bins));try! stdout.flush();}

        if (bins <= sBound) {

        }
        else if (bins <= mBound) {

        }
        else if (bins <= lBound) {
        
        }
        else {

        }

        return iv;
    }
    
}