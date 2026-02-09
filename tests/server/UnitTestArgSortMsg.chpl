prototype module UnitTestArgSort
{
    config const NVALS = 2**13;
    config const LEN = 2**20;
    use TestBase;
    
    use RandMsg;
    use IndexingMsg;
    use ReductionMsg;
    use ArgSortMsg;

    // unit test for ArgSortMsg
    proc main() {
        writeln("Unit Test for ArgSortMsg");
        var st = new owned SymTab();

        var reqMsg: string;
        var rep_msg: string;

        // create an array filled with random int64 returned in symbol table
        var aname = nameForRandintMsg(LEN, DType.Int64, 0, NVALS, st);

        // sort it and return iv in symbol table
        var cmd = "argsort";
        reqMsg = try! "%s pdarray %s".format(ArgSortMsg.SortingAlgorithm.RadixSortLSD: string, aname);
        writeReq(reqMsg);
        var d: Diags;
        d.start();
        rep_msg = argsortMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("argsortMsg");
        writeRep(rep_msg);

        // apply iv to pdarray return sorted array
        cmd = "[pdarray]";
        var ivname = parseName(rep_msg); // get name from argsort reply msg
        reqMsg = try! "%s %s".format(aname, ivname);
        writeReq(reqMsg);
        d.start();
        rep_msg = pdarrayIndexMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("pdarrayIndexMsg");
        writeRep(rep_msg);

        // check for result to be sorted
        writeln("Checking that arrays are sorted");
        cmd = "reduction";
        var subCmd = "is_sorted";
        var bname = parseName(rep_msg); // get name from [pdarray] reply msg
        reqMsg = try! "%s %s".format(subCmd, bname);
        writeReq(reqMsg);
        d.start();
        rep_msg = reductionMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("reductionMsg");
        writeln("ANSWER >>> ",rep_msg," <<<");

        // create a second array filled with random float64 returned in symbol table
        var fname = nameForRandintMsg(LEN, DType.Float64, 0, 1, st);

        // cosort both int and real arrays and return iv in symbol table
        cmd = "coargsort";
        reqMsg = try! "%s %i %s %s pdarray pdarray".format(ArgSortMsg.SortingAlgorithm.RadixSortLSD: string, 2, aname, fname);
        writeReq(reqMsg);
        d.start();
        rep_msg = coargsortMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("coargsortMsg");
        writeRep(rep_msg);

        // apply iv to pdarray return sorted array
        cmd = "[pdarray]";
        var coivname = parseName(rep_msg); // get name from argsort reply msg
        reqMsg = try! "%s %s".format(aname, coivname);
        writeReq(reqMsg);
        d.start();
        rep_msg = pdarrayIndexMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("pdarrayIndexMsg");
        writeRep(rep_msg);
        var coaname = parseName(rep_msg);
        reqMsg = try! "%s %s".format(fname, coivname);
        writeReq(reqMsg);
        d.start();
        rep_msg = pdarrayIndexMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("pdarrayIndexMsg");
        writeRep(rep_msg);
        var cofname = parseName(rep_msg);
        writeln("Checking that arrays are cosorted");
        var coa = toSymEntry(toGenSymEntry(st.lookup(coaname)), int);
        var cof = toSymEntry(toGenSymEntry(st.lookup(cofname)), real);
        var allSorted: atomic bool = true;
        forall (a, f, i) in zip(coa.a[1..], cof.a[1..], coa.a.domain.low+1..) {
          ref coaa = coa.a;
          ref cofa = cof.a;
          if (coaa[i-1] > a) {
            allSorted.write(false);
          } else if (coaa[i-1] == a) {
            if (cofa[i-1] > f) {
              allSorted.write(false);
            }
            // if cofa[i-1] <= f continue
          }
          // if coaa[i-1] < a continue
        }
        writeln("ANSWER >>> ", allSorted, " <<<");
    }

}

