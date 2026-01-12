prototype module UnitTestArgSort
{
    config const NVALS = 2**13;
    config const LEN = 2**20;
    config const filename = "UnitTestArgSort.array";

    use TestBase;

    use RandMsg;
    use IndexingMsg;
    use ReductionMsg;
    use ArgSortMsg;

    proc writeIntArray(a:[?D] int, filename:string) {
        var f: open(filename, ioMode.cw).type;
        try { f = open(filename, ioMode.cw); } catch e { exit(1); }
        var w: f.writer(kind=ionative).type ;
        try { w = f.writer(kind=ionative); } catch e { exit(1); }
      try! w.write(D.size);
      try! w.write(a);
      try! w.close();
      try! f.close();
    }
    
    proc main() {
        writeln("Unit Test for localArgSortMsg");
        var st = new owned SymTab();

        var reqMsg: string;
        var rep_msg: string;

        // create an array filled with random int64 returned in symbol table
        var aname = nameForRandintMsg(LEN, DType.Int64, 0, NVALS, st);

        // sort it and return iv in symbol table
        var cmd = "localArgsort";
        var orig = toSymEntry(st.lookup(aname), int);
        writeIntArray(orig.a, filename+".original");
        reqMsg = try! "%s".format(aname);
        var d: Diags;
        d.start();
        rep_msg = localArgsortMsg(cmd=cmd, payload=reqMsg.encode(), st);
        d.stop("localArgsortMsg");
        writeRep(rep_msg);

        // apply iv to pdarray return sorted array
        cmd = "[pdarray]";
        var ivname = parseName(rep_msg); // get name from argsort reply msg
        var iv = toSymEntry(st.lookup(ivname), int);
        writeIntArray(iv.a, filename+".permutation");
        reqMsg = try! "%s %s".format(aname, ivname);
        d.start();
        rep_msg = pdarrayIndexMsg(cmd=cmd, payload=reqMsg.encode(), st);
        d.stop("pdarrayIndexMsg");
        writeRep(rep_msg);

        // check for result to be sorted
        cmd = "reduction";
        var subCmd = "is_locally_sorted";
        var bname = parseName(rep_msg); // get name from [pdarray] reply msg
        var locSorted = toSymEntry(st.lookup(bname), int);
        writeIntArray(locSorted.a, filename+".locally_sorted");
        reqMsg = try! "%s %s".format(subCmd, bname);
        d.start();
        rep_msg = reductionMsg(cmd=cmd, payload=reqMsg.encode(), st);
        d.stop("reductionMsg");
        writeln("ANSWER >>> ",rep_msg," <<<");
    }

}

