prototype module efuncTest
{
    config const NVALS = 2**13;
    config const LEN = 2**20;
    use TestBase;
    
    use MsgProcessing;

    proc main() {
        writeln("Unit Test for akpow");
        var st = new owned SymTab();

        var reqMsg: string;
        var repMsg: string;
        var repMsg2: string;

        var cmd = "create";
        var len = 5;
        var dtype = DType.Int64;
        reqMsg = try! "%s %s %i".format(cmd, dtype2str(dtype), len);
        var t1 = Time.getCurrentTime();
        repMsg = createMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln(repMsg);

        cmd = "set";
        var name = parseName(repMsg);
        dtype = DType.Int64;
        var v2= 2;
        reqMsg = try! "%s %s %s %i".format(cmd, name, dtype2str(dtype), v2);
        t1 = Time.getCurrentTime();
        repMsg = setMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln(repMsg);

        var aname = parseName(repMsg);

        cmd = "create";
        len = 5;
        dtype = DType.Int64;
        reqMsg = try! "%s %s %i".format(cmd, dtype2str(dtype), len);
        t1 = Time.getCurrentTime();
        repMsg = createMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln(repMsg);

        cmd = "set";
        name = parseName(repMsg);
        dtype = DType.Int64;
        v2= -2;
        reqMsg = try! "%s %s %s %i".format(cmd, name, dtype2str(dtype), v2);
        t1 = Time.getCurrentTime();
        repMsg = setMsg(reqMsg, st);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        writeln(repMsg);
        
        var bname = parseName(repMsg);

        cmd = "opeqvv";
        var op = "**=";
        
        dtype = DType.Int64;
        var value=-1;
        reqMsg = try! "%s %s %s %s".format(cmd, op, aname,bname);
        t1 = Time.getCurrentTime();
        repMsg = opeqvvMsg(reqMsg, st);
        writeln(repMsg);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        st.pretty();
    }
}
