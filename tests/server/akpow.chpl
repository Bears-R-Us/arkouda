prototype module efuncTest
{
    config const NVALS = 2**13;
    config const LEN = 2**20;
    use TestBase;
    
    use OperatorMsg;
    use MsgProcessing;

    proc main() {
        writeln("Unit Test for akpow");
        var st = new owned SymTab();

        var reqMsg: string;
        var rep_msg: string;

        var cmd = "create";
        var len = 5;
        var dtype = DType.Int64;
        reqMsg = try! "%s %i".format(dtype2str(dtype), len);
        var d: Diags;
        d.start();
        rep_msg = createMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("createMsg");
        writeRep(rep_msg);

        cmd = "set";
        var name = parseName(rep_msg);
        dtype = DType.Int64;
        var v2= 2;
        reqMsg = try! "%s %s %i".format(name, dtype2str(dtype), v2);
        d.start();
        rep_msg = setMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("setMsg");
        writeRep(rep_msg);

        var aname = parseName(rep_msg);

        cmd = "create";
        len = 5;
        dtype = DType.Int64;
        reqMsg = try! "%s %i".format(dtype2str(dtype), len);
        d.start();
        rep_msg = createMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("createMsg");
        writeRep(rep_msg);

        cmd = "set";
        name = parseName(rep_msg);
        dtype = DType.Int64;
        v2= -2;
        reqMsg = try! "%s %s %i".format(name, dtype2str(dtype), v2);
        d.start();
        rep_msg = setMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("setMsg");
        writeRep(rep_msg);
        
        var bname = parseName(rep_msg);

        cmd = "opeqvv";
        var op = "**=";
        
        dtype = DType.Int64;
        var value=-1;
        reqMsg = try! "%s %s %s".format(op, aname,bname);
        d.start();
        rep_msg = opeqvvMsg(cmd=cmd, payload=reqMsg, st).msg;
        d.stop("opeqvvMsg");
        writeRep(rep_msg);
    }
}
