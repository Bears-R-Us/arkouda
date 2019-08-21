module efuncTest
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

    use MsgProcessing;
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
        var dtype = DType.Float64;
        reqMsg = try! "%s %i %i %i %s".format(cmd, aMin, aMax, len, dtype2str(dtype));
        var t1 = Time.getCurrentTime();
        repMsg = randintMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln(repMsg);

        // sort it and return iv in symbol table
        cmd = "efunc";
        var op = "sin";
        var aname = parseName(repMsg); // get name from randint reply msg
        reqMsg = try! "%s %s %s".format(cmd, op, aname);
        t1 = Time.getCurrentTime();
        repMsg = efuncMsg(reqMsg, st);
        writeln(repMsg);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        // check for result to be sorted
        cmd = "reduction";
        var subCmd = "sum";
        var bname = parseName(repMsg); // get name from [pdarray] reply msg
        reqMsg = try! "%s %s %s".format(cmd, subCmd, bname);
        t1 = Time.getCurrentTime();
        repMsg = reductionMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln("ANSWER >>> ",repMsg," <<<");
    }

}

