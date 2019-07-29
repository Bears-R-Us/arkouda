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
        var repMsg2: string;

        //create an array filled with random int64 returned in symbol table
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

        //sort it and return iv in symbol table
        cmd = "opeqvv";
        var op = "**=";
         // get name from randint reply msg
        
        dtype = DType.Int64;
        var value=-1;
        reqMsg = try! "%s %s %s %s".format(cmd, op, aname,bname);
        t1 = Time.getCurrentTime();
        repMsg = opeqvvMsg(reqMsg, st);
        writeln(repMsg);
        writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        // // check for result to be sorted
        // cmd = "reduction";
        // var subCmd = "sum";
        // var bname = parseName(repMsg); // get name from [pdarray] reply msg
        // reqMsg = try! "%s %s %s".format(cmd, subCmd, bname);
        // t1 = Time.getCurrentTime();
        // repMsg = reductionMsg(reqMsg, st);
        // writeln("time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
        // writeln("ANSWER >>> ",repMsg," <<<");




        // writeln("pt2: custom array. use binopvs **");
        // var myLen =5;
        // var rname = st.nextName();
        // st.addEntry(rname,myLen,int);

        // var aD = makeDistDom(myLen);
        // var a = makeDistArray(myLen, int);
        // st.addEntry(rname, new shared SymEntry(a));

        // var symArr=st.lookup(rname);


        // for i in symArr2.a.domain{
        //     symArr.a[i]=i+1;
        // }
        // writeln(st.info(rname));
        // writeln(st.datastr(rname,6));
        // st.pretty();

        // st.addEntry(rname, symArr2);
    
        st.pretty();
    }




}

