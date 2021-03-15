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
        var repMsg: string;

        // create an array filled with random int64 returned in symbol table
        var aname = nameForRandintMsg(LEN, DType.Int64, 0, NVALS, st);

        // sort it and return iv in symbol table
        var cmd = "argsort";
        reqMsg = try! "pdarray %s".format(aname);
        writeReq(reqMsg);
        var d: Diags;
        d.start();
        repMsg = argsortMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("argsortMsg");
        writeRep(repMsg);

        // apply iv to pdarray return sorted array
        cmd = "[pdarray]";
        var ivname = parseName(repMsg); // get name from argsort reply msg
        reqMsg = try! "%s %s".format(aname, ivname);
        writeReq(reqMsg);
        d.start();
        repMsg = pdarrayIndexMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("pdarrayIndexMsg");
        writeRep(repMsg);

        // check for result to be sorted
        writeln("Checking that arrays are sorted");
        cmd = "reduction";
        var subCmd = "is_sorted";
        var bname = parseName(repMsg); // get name from [pdarray] reply msg
        reqMsg = try! "%s %s".format(subCmd, bname);
        writeReq(reqMsg);
        d.start();
        repMsg = reductionMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("reductionMsg");
        writeln("ANSWER >>> ",repMsg," <<<");

        // create a second array filled with random float64 returned in symbol table
        var fname = nameForRandintMsg(LEN, DType.Float64, 0, 1, st);

        // cosort both int and real arrays and return iv in symbol table
        cmd = "coargsort";
        reqMsg = try! "%i %s %s pdarray pdarray".format(2, aname, fname);
        writeReq(reqMsg);
        d.start();
        repMsg = coargsortMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("coargsortMsg");
        writeRep(repMsg);

        // apply iv to pdarray return sorted array
        cmd = "[pdarray]";
        var coivname = parseName(repMsg); // get name from argsort reply msg
        reqMsg = try! "%s %s".format(aname, coivname);
        writeReq(reqMsg);
        d.start();
        repMsg = pdarrayIndexMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("pdarrayIndexMsg");
        writeRep(repMsg);
        var coaname = parseName(repMsg);
        reqMsg = try! "%s %s".format(fname, coivname);
        writeReq(reqMsg);
        d.start();
        repMsg = pdarrayIndexMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("pdarrayIndexMsg");
        writeRep(repMsg);
        var cofname = parseName(repMsg);
        writeln("Checking that arrays are cosorted");
        var coa = toSymEntry(st.lookup(coaname), int);
        var cof = toSymEntry(st.lookup(cofname), real);
        var allSorted: atomic bool = true;
        forall (a, f, i) in zip(coa.a[1..], cof.a[1..], coa.aD.low+1..) {
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

