prototype module efuncTest
{
    config const NVALS = 2**13;
    config const LEN = 2**20;
    use TestBase;

    use MsgProcessing;

    proc main() {
        writeln("Unit Test for efuncMsg");
        var st = new owned SymTab();

        var reqMsg: string;
        var repMsg: string;

        // create an array filled with random float64
        var aname = nameForRandintMsg(LEN, DType.Float64, 0, NVALS, st);

        var cmd = "efunc";
        var op = "sin";
        reqMsg = try! "%s %s %s".format(cmd, op, aname);
        var t1 = Time.getCurrentTime();
        repMsg = efuncMsg(reqMsg, st);
        writeln(repMsg);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        // check for result
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
