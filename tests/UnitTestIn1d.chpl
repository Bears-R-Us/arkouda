module UnitTestIn1d
{
    use In1d;
    
    use Random;
    use Time only;

    use BlockDist;


    // fill a with integers from interval aMin..(aMax-1)
    proc fillRandInt(a: [?aD] int, aMin: int, aMax: int) {

        var t1 = Time.getCurrentTime();
        
        coforall loc in Locales {
            on loc {
                var R = new owned RandomStream(real); R.getNext();
                [i in a.localSubdomain()] a[i] = (R.getNext() * (aMax -aMin) + aMin):int;
            }
        }

        writeln("compute time = ", Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
    }

    proc test_in1d(n: int, m: int, nVals: int) {

        writeln((n, m, nVals)); try! stdout.flush();

        const aDom = newBlockDom({0..#n});

        var a: [aDom] int;

        var aMin = 0;
        var aMax = nVals-1;

        // fill a with random ints from a range
        fillRandInt(a, aMin, aMax+1);

        const bDom = newBlockDom({0..#m});
        var b: [bDom] int;
        var expected: real;

        if (m < nVals) {
            [(i, bi) in zip(bDom, b)] bi = i * (nVals / m);
            expected = n:real * (m:real / nVals:real);
        }
        else {
            [(i, bi) in zip(bDom, b)] bi = i % nVals;
            expected = n:real;
        }

        writeln(">>> in1d");
        var t1 = Time.getCurrentTime();
        // returns a boolean vector
        var truth = in1dAr2PerLocAssoc(a, b);
        writeln("total time = ", Time.getCurrentTime() - t1, "sec"); try! stdout.flush();
        writeln("<<< #a[i] in b = ", + reduce truth, " (expected ", expected, ")");try! stdout.flush();
    }

    config const N = 1_000_000;
    config const M = 1_000;
    config const NVALS = 1_000_000;

    proc main() {
        test_in1d(N, M, NVALS);
    }
    
}
