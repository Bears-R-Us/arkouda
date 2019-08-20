module UnitTestArgSort
{
    config const NVALS = 2**13;
    config const LEN = 2**20;
    config const filename = "UnitTestArgSort.array";
    use ServerConfig;
    
    use Time only;
    use Math only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use AryUtil;

    use RandMsg;
    use IndexingMsg;
    use ReductionMsg;
    
    // module to be unit tested
    use ArgSortMsg;

    proc parseName(s: string): string {
        var fields = s.split(); 
        return fields[2];
    }

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
    
    // unit test for localArgSortMsg
    proc main() {
        writeln("Unit Test for localArgSortMsg");
        var st = new owned SymTab();

        var reqMsg: string;
        var repMsg: string;

        // create an array filled with random int64 returned in symbol table
        var cmd = "randint";
        var aMin = 0;
        var aMax = NVALS;
        var len = LEN;
        var dtype = DType.Int64;
        reqMsg = try! "%s %i %i %i %s".format(cmd, aMin, aMax, len, dtype2str(dtype));
        var t1 = Time.getCurrentTime();
        repMsg = randintMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln(repMsg);

        // sort it and return iv in symbol table
        cmd = "localArgsort";
        var aname = parseName(repMsg); // get name from randint reply msg
	var orig = toSymEntry(st.lookup(aname), int);
	writeIntArray(orig.a, filename+".original");
        reqMsg = try! "%s %s".format(cmd, aname);
        t1 = Time.getCurrentTime();
        repMsg = localArgsortMsg(reqMsg, st);
        writeln(repMsg);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        // apply iv to pdarray return sorted array
        cmd = "[pdarray]";
        var ivname = parseName(repMsg); // get name from argsort reply msg
	var iv = toSymEntry(st.lookup(ivname), int);
	writeIntArray(iv.a, filename+".permutation");
        reqMsg = try! "%s %s %s".format(cmd, aname, ivname);
        t1 = Time.getCurrentTime();
        repMsg = pdarrayIndexMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln(repMsg);

        // check for result to be sorted
        cmd = "reduction";
        var subCmd = "is_locally_sorted";
        var bname = parseName(repMsg); // get name from [pdarray] reply msg
	var locSorted = toSymEntry(st.lookup(bname), int);
	writeIntArray(locSorted.a, filename+".locally_sorted");
        reqMsg = try! "%s %s %s".format(cmd, subCmd, bname);
        t1 = Time.getCurrentTime();
        repMsg = reductionMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln("ANSWER >>> ",repMsg," <<<");
    }

}

