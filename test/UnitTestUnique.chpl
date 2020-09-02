prototype module UnitTestUnique
{
    use TestBase;

    use Unique;
    
    use Random;


    // fill a with integers from interval aMin..(aMax-1)
    proc fillRandInt(a: [?aD] ?t, aMin: t, aMax: t) {
      var d: Diags; d.start();
      coforall loc in Locales {
        on loc {
          var R = new owned RandomStream(real); R.getNext();
          for i in a.localSubdomain() { a[i] = (R.getNext() * (aMax -aMin) + aMin):t; }
        }
      }
      d.stop("fillRandInt");
    }

    // fill a with integers from interval aMin..(aMax-1)
    proc fillRandFromArray(a: [?aD] int, b: [?bD] int) {
      var d: Diags; d.start();
      coforall loc in Locales {
        on loc {
          var R = new owned RandomStream(real); R.getNext();
          for i in a.localSubdomain() {
            var vi = (R.getNext() * ((bD.high+1) - bD.low) + bD.low):int;//is this right????
            a[i] = b[vi];
          }
        }
      }
      d.stop("fillRandFromArray");
    }

    proc createRandomStrings(n: int, minLen: int, maxLen: int, nUnique: int, st: borrowed SymTab) throws {
      var d: Diags;
      d.start();
      var uLens = makeDistArray(nUnique, int);
      // Add 1 for null byte
      fillRandInt(uLens, minLen+1, maxLen+1);
      const uBytes = + reduce uLens;
      var uSegs = (+ scan uLens) - uLens;
      var uVals = makeDistArray(uBytes, uint(8));
      fillRandInt(uVals, 32:uint(8), 126:uint(8)); // printable ascii
      // Terminate with null bytes
      [(s, l) in zip(uSegs, uLens)] uVals[s+l-1] = 0:uint(8);
      var uStr = new owned SegString(uSegs, uVals, st);
      // Indices to sample from unique strings
      var inds = makeDistArray(n, int);
      fillRandInt(inds, 0, nUnique - 1);
      var (segs, vals) = uStr[inds];
      var str = new shared SegString(segs, vals, st);
      d.stop("createRandomStrings");
      writeSegString("str", str);
      return str;
    }

    proc strFromArrays(segs: [] int, vals: [] uint(8), st: borrowed SymTab) throws {
      var segName = st.nextName();
      var valName = st.nextName();
      var segEntry = new shared SymEntry(segs);
      var valEntry = new shared SymEntry(vals);
      st.addEntry(segName, segEntry);
      st.addEntry(valName, valEntry);
      var str = new shared SegString(segName, valName, st);
      return str;
    }

    // n: length of array
    // aMin: min value
    // aMax: max value
    // nValues: number of unique values
    
    proc test_unique(n: int, aMin: int, aMax: int, nVals: int) {
        const aDom = makeDistDom(n);

        var a: [aDom] int;

        const valsDom = makeDistDom(nVals);
        var vals: [valsDom] int;
        
        // fill a with random ints from a range
        fillRandInt(vals, aMin, aMax+1);
        fillRandFromArray(a,vals);
        
        writeln(">>> uniqueSort");
        var d: Diags;

        d.start();
        var (aV1,aC1) = uniqueSort(a);
        d.stop("uniqueSort");

        //printAry("aV1 = ",aV1);
        //printAry("aC1 = ",aC1);

        //writeln("aV1.size = ",aV1.size);
        //writeln("totalCounts = ",+ reduce aC1); try! stdout.flush();
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

    proc test_strings(n: int, minLen: int, maxLen: int, nVals: int) throws {
        var st = new owned SymTab();
        var str = createRandomStrings(n, minLen, maxLen, nVals, st);
        
        writeln(">>> uniqueGroup");
        var d: Diags;
        d.start();
        var (uo1, uv1, c1, inv1) = uniqueGroup(str);
        d.stop("uniqueGroup");

        // var uStr1 = strFromArrays(uo1, uv1, st);
        var uStr1 = new owned SegString(uo1, uv1, st);
        writeSegString("Unique strings:", uStr1);
        
        //writeln("uStr1.size = ",uStr1.size);
        //writeln("totalCounts = ",+ reduce c1); try! stdout.flush();
        writeln("testing return_inverse...");
        var (uo2, uv2, c2, inv2) = uniqueGroup(str, returnInverse=true);
        // var uStr2 = strFromArrays(uo2, uv2, st);
        var uStr2 = new owned SegString(uo2, uv2, st);
        writeln("offsets, values, and counts match? >>> ", (&& reduce (uo2 == uo1)) && (&& reduce (uv2 == uv1)) && (&& reduce (c2 == c1)), " <<<" );
        var (rtSegs, rtVals) = uStr2[inv2];
        // var roundTrip = strFromArrays(rtSegs, rtVals, st);
        var roundTrip = new owned SegString(rtSegs, rtVals, st);
        writeln("original array correctly reconstructed from inverse? >>> ", && reduce (str == roundTrip), " <<<");
    }

    config const N = 10_000; // length of array
    config const AMIN = 1_000; // min value
    config const AMAX = 30_000_000_000; // max value
    config const NVALS = 1_000; // number of unique values
    config const testStrings = true;
    config const MINLEN = 0;
    config const MAXLEN = 10;

    proc main() {
        test_unique(N, AMIN, AMAX, NVALS);
        if testStrings {
          try! test_strings(N, MINLEN, MAXLEN, NVALS);
        }
    }
    
}
