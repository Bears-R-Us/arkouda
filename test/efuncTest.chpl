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
        reqMsg = try! "%s %s".format(op, aname);
        var d: Diags;
        d.start();
        repMsg = efuncMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("efuncMsg");
        writeRep(repMsg);

        // check for result
        cmd = "reduction";
        var subCmd = "sum";
        var bname = parseName(repMsg); // get name from [pdarray] reply msg
        reqMsg = try! "%s %s".format(subCmd, bname);
        d.start();
        repMsg = reductionMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("reductionMsg");
        //writeln("ANSWER >>> ",repMsg," <<<"); TODO 
    }
}
