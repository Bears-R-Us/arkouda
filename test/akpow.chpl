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

        var cmd = "create";
        var len = 5;
        var dtype = DType.Int64;
        reqMsg = try! "%s %s %i".format(cmd, dtype2str(dtype), len);
        var d: Diags;
        d.start();
        repMsg = createMsg(reqMsg, st);
        d.stop("createMsg");
        writeRep(repMsg);

        cmd = "set";
        var name = parseName(repMsg);
        dtype = DType.Int64;
        var v2= 2;
        reqMsg = try! "%s %s %s %i".format(cmd, name, dtype2str(dtype), v2);
        d.start();
        repMsg = setMsg(reqMsg, st);
        d.stop("setMsg");
        writeRep(repMsg);

        var aname = parseName(repMsg);

        cmd = "create";
        len = 5;
        dtype = DType.Int64;
        reqMsg = try! "%s %s %i".format(cmd, dtype2str(dtype), len);
        d.start();
        repMsg = createMsg(reqMsg, st);
        d.stop("createMsg");
        writeRep(repMsg);

        cmd = "set";
        name = parseName(repMsg);
        dtype = DType.Int64;
        v2= -2;
        reqMsg = try! "%s %s %s %i".format(cmd, name, dtype2str(dtype), v2);
        d.start();
        repMsg = setMsg(reqMsg, st);
        d.stop("setMsg");
        writeRep(repMsg);
        
        var bname = parseName(repMsg);

        cmd = "opeqvv";
        var op = "**=";
        
        dtype = DType.Int64;
        var value=-1;
        reqMsg = try! "%s %s %s %s".format(cmd, op, aname,bname);
        d.start();
        repMsg = opeqvvMsg(reqMsg, st);
        d.stop("opeqvvMsg");
        writeRep(repMsg);
        st.pretty();
    }
}
