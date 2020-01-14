module UnitTestUnique
{
    use Unique;
    
    use Random;
    use Time only;

    use BlockDist;
    use AryUtil;


    // fill a with integers from interval aMin..(aMax-1)
    proc fillRandInt(a: [?aD] int, aMin: int, aMax: int) {

        var t1 = Time.getCurrentTime();
        
        coforall loc in Locales {
            on loc {
                var R = new owned RandomStream(real); R.getNext();
                for i in a.localSubdomain() { a[i] = (R.getNext() * (aMax -aMin) + aMin):int; }
            }
        }

        writeln("compute time = ", Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
    }

    // fill a with integers from interval aMin..(aMax-1)
    proc fillRandFromArray(a: [?aD] int, b: [?bD] int) {

        var t1 = Time.getCurrentTime();
        
        coforall loc in Locales {
            on loc {
                var R = new owned RandomStream(real); R.getNext();
                for i in a.localSubdomain() {
                    var vi = (R.getNext() * ((bD.high+1) - bD.low) + bD.low):int;//is this right????
                    a[i] = b[vi];
                }
            }
        }

        writeln("compute time = ", Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
    }

    // n: length of array
    // aMin: min value
    // aMax: max value
    // nValues: number of unique values
    
    proc test_unique(n: int, aMin: int, aMax: int, nVals: int) {

        writeln("n = ",n);
        writeln("aMin = ", aMin);
        writeln("aMax = ", aMax);
        writeln("nVals = ",nVals);
        try! stdout.flush();

        const aDom = newBlockDom({0..#n});

        var a: [aDom] int;

        const valsDom = newBlockDom({0..#nVals});
        var vals: [valsDom] int;
        
        // fill a with random ints from a range
        fillRandInt(vals, aMin, aMax+1);
        fillRandFromArray(a,vals);
        
        writeln(">>> uniqueSort");
        var t1 = Time.getCurrentTime();

        var (aV1,aC1) = uniqueSort(a);

        writeln("total time = ", Time.getCurrentTime() - t1, "sec"); try! stdout.flush();

        printAry("aV1 = ",aV1);
        printAry("aC1 = ",aC1);

        writeln("aV1.size = ",aV1.size);
        writeln("totalCounts = ",+ reduce aC1); try! stdout.flush();
        var present: [aDom] bool;
        forall (x, p) in zip(a, present) {
          for u in aV1 {
            if (x == u) {
              p = true;
              break;
            }
          }
        }
        writeln("all values present? >>> ", && reduce present, " <<<"); try! stdout.flush();
        writeln("testing return_inverse...");
        var (aV2, aC2, inv) = uniqueSortWithInverse(a);
        writeln("values and counts match? >>> ", (&& reduce (aV2 == aV1)) && (&& reduce (aC2 == aC1)), " <<<" );
        var a2: [aDom] int;
        forall (x, idx) in zip(a2, inv) {
            x = aV2[idx];
        }
        writeln("original array correctly reconstructed from inverse? >>> ", && reduce (a == a2), " <<<");
    }

    config const N = 1_000_000; // length of array
    config const AMIN = 1_000; // min value
    config const AMAX = 30_000_000_000; // max value
    config const NVALS = 1_000; // number of unique values

    proc main() {
        test_unique(N, AMIN, AMAX, NVALS);
    }
    
}
