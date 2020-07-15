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
        var f: open(filename, iomode.cw).type;
        try { f = open(filename, iomode.cw); } catch e { exit(1); }
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
        var repMsg: string;

        // create an array filled with random int64 returned in symbol table
        var aname = nameForRandintMsg(LEN, DType.Int64, 0, NVALS, st);

        // sort it and return iv in symbol table
        var cmd = "localArgsort";
        var orig = toSymEntry(st.lookup(aname), int);
        writeIntArray(orig.a, filename+".original");
        reqMsg = try! "%s".format(aname);
        var d: Diags;
        d.start();
        repMsg = localArgsortMsg(cmd=cmd, payload=reqMsg.encode(), st);
        d.stop("localArgsortMsg");
        writeRep(repMsg);

        // apply iv to pdarray return sorted array
        cmd = "[pdarray]";
        var ivname = parseName(repMsg); // get name from argsort reply msg
        var iv = toSymEntry(st.lookup(ivname), int);
        writeIntArray(iv.a, filename+".permutation");
        reqMsg = try! "%s %s".format(aname, ivname);
        d.start();
        repMsg = pdarrayIndexMsg(cmd=cmd, payload=reqMsg.encode(), st);
        d.stop("pdarrayIndexMsg");
        writeRep(repMsg);

        // check for result to be sorted
        cmd = "reduction";
        var subCmd = "is_locally_sorted";
        var bname = parseName(repMsg); // get name from [pdarray] reply msg
        var locSorted = toSymEntry(st.lookup(bname), int);
        writeIntArray(locSorted.a, filename+".locally_sorted");
        reqMsg = try! "%s %s".format(subCmd, bname);
        d.start();
        repMsg = reductionMsg(cmd=cmd, payload=reqMsg.encode(), st);
        d.stop("reductionMsg");
        writeln("ANSWER >>> ",repMsg," <<<");
    }

}

