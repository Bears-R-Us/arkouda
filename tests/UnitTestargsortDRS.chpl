module UnitTestargsortDRS
{
    use ArgsortDRS;
    
    // fill a with integers from interval aMin..(aMax-1)
    proc fillRandInt(a: [?aD] int, aMin: int, aMax: int) {
    
        var t1 = Time.getCurrentTime();
        coforall loc in Locales {
            on loc {
                var R = new owned RandomStream(real); R.getNext();
                [i in a.localSubdomain()] a[i] = (R.getNext() * (aMax - aMin) + aMin):int;
            }
        }
        writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
    
    }
    proc isSorted(A:[?D] ?t): bool {
        var sorted: bool;
        sorted = true;
        forall (a,i) in zip(A,D) with (&& reduce sorted) {
            if i > D.low {
                sorted &&= (A[i-1] <= a);
            }
        }
        return sorted;
    }

    proc test_argsort(n: int, nVals: int) {
        const blockDom = newBlockDom({0..#n});
        var a: [blockDom] int;

        var aMin = 0;
        var aMax = nVals-1;
        writeln((n, nVals));
        // fill an array with random ints from a range
        fillRandInt(a, aMin, aMax+1);

        var localCopy:[0..#n] int = a;

        writeln(">>> argsortDRS");
        var t1 = Time.getCurrentTime();
        // returns a perm vector
        var iv = argsortDRS(a, aMin, aMax);
        writeln("sort time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        // permute into sorted order
        var sorted: [a.domain] int = a[iv];

        // check to see if we successfully sorted a
        writeln("<<< isSorted = ", isSorted(sorted));
    }

    config const N = 1000;
    config const NVALS = 8192;

    proc main() {
        test_argsort(N, NVALS);
    }

}
