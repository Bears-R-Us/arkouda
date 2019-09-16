module UnitTestArgSort
{
    config const NVALS = 2**13;
    config const LEN = 2**20;
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
    
    // unit test for ArgSortMsg
    proc main() {
        writeln("Unit Test for ArgSortMsg");
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
        cmd = "argsort";
        var aname = parseName(repMsg); // get name from randint reply msg
        reqMsg = try! "%s %s".format(cmd, aname);
	writeln(reqMsg);
        t1 = Time.getCurrentTime();
        repMsg = argsortMsg(reqMsg, st);
        writeln(repMsg);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        // apply iv to pdarray return sorted array
        cmd = "[pdarray]";
        var ivname = parseName(repMsg); // get name from argsort reply msg
        reqMsg = try! "%s %s %s".format(cmd, aname, ivname);
	writeln(reqMsg);
        t1 = Time.getCurrentTime();
        repMsg = pdarrayIndexMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln(repMsg);

        // check for result to be sorted
        cmd = "reduction";
        var subCmd = "is_sorted";
        var bname = parseName(repMsg); // get name from [pdarray] reply msg
        reqMsg = try! "%s %s %s".format(cmd, subCmd, bname);
	writeln(reqMsg);
        t1 = Time.getCurrentTime();
        repMsg = reductionMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln("ANSWER >>> ",repMsg," <<<");

	// create a second array filled with random float64 returned in symbol table
        cmd = "randint";
        len = LEN;
        dtype = DType.Float64;
        reqMsg = try! "%s %i %i %i %s".format(cmd, 0, 1, len, dtype2str(dtype));
	writeln(reqMsg);
        t1 = Time.getCurrentTime();
        repMsg = randintMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln(repMsg);

	// cosort both int and real arrays and return iv in symbol table
        cmd = "coargsort";
        var fname = parseName(repMsg); // get name from randint reply msg
        reqMsg = try! "%s %i %s %s".format(cmd, 2, aname, fname);
	writeln(reqMsg);
        t1 = Time.getCurrentTime();
        repMsg = coargsortMsg(reqMsg, st);
        writeln(repMsg);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

	// apply iv to pdarray return sorted array
        cmd = "[pdarray]";
        var coivname = parseName(repMsg); // get name from argsort reply msg
        reqMsg = try! "%s %s %s".format(cmd, aname, coivname);
	writeln(reqMsg);
        t1 = Time.getCurrentTime();
        repMsg = pdarrayIndexMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln(repMsg);
	var coaname = parseName(repMsg);
	reqMsg = try! "%s %s %s".format(cmd, fname, coivname);
	writeln(reqMsg);
	t1 = Time.getCurrentTime();
        repMsg = pdarrayIndexMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln(repMsg);
	var cofname = parseName(repMsg);
	writeln("Checking that arrays are cosorted");
	var coa = toSymEntry(st.lookup(coaname), int);
	var cof = toSymEntry(st.lookup(cofname), real);
	var allSorted = true;
	forall (a, f, i) in zip(coa.a[1..], cof.a[1..], coa.aD.low+1..) with (ref allSorted) {
	  ref coaa = coa.a;
	  ref cofa = cof.a;
	  if (coaa[i-1] > a) {
	    allSorted = false;
	  } else if (coaa[i-1] == a) {
	    if (cofa[i-1] > f) {
	      allSorted = false;
	    }
	    // if cofa[i-1] <= f continue
	  }
	  // if coaa[i-1] < a continue
	}
	writeln("ANSWER >>>", allSorted, "<<<");
    }

}

