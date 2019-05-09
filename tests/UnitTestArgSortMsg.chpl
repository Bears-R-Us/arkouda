module UnitTestArgSort
{
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
        var aMax = 2**13;
        var len = 2**20;
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
        t1 = Time.getCurrentTime();
        repMsg = argsortMsg(reqMsg, st);
        writeln(repMsg);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        // apply iv to pdarray return sorted array
        cmd = "[pdarray]";
        var ivname = parseName(repMsg); // get name from argsort reply msg
        reqMsg = try! "%s %s %s".format(cmd, aname, ivname);
        t1 = Time.getCurrentTime();
        repMsg = pdarrayIndexMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln(repMsg);

        // check for result to be sorted
        cmd = "reduction";
        var subCmd = "is_sorted";
        var bname = parseName(repMsg); // get name from [pdarray] reply msg
        reqMsg = try! "%s %s %s".format(cmd, subCmd, bname);
        t1 = Time.getCurrentTime();
        repMsg = reductionMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln("ANSWER >>> ",repMsg," <<<");
    }

}

