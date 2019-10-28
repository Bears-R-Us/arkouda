
module MsgProcessing
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use AryUtil;
    
    use OperatorMsg;
    use RandMsg;
    use IndexingMsg;
    use UniqueMsg;
    use In1dMsg;
    use HistogramMsg;
    use ArgSortMsg;
    use ReductionMsg;
    use FindSegmentsMsg;
    use EfuncMsg;
    use ConcatenateMsg;
    
    /* 
    Parse, execute, and respond to a create message 

    :arg reqMsg: request containing (cmd,dtype,size)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) response message
    */
    proc createMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var dtype = str2dtype(fields[2]);
        var size = try! fields[3]:int;

        // get next symbol name
        var rname = st.nextName();
        
        // if verbose print action
        if v {try! writeln("%s %s %i : %s".format(cmd,dtype2str(dtype),size,rname)); try! stdout.flush();}
        // create and add entry to symbol table
        st.addEntry(rname, size, dtype);
        // response message
        return try! "created " + st.attrib(rname);
    }

    /* 
    Parse, execute, and respond to a delete message 

    :arg reqMsg: request containing (cmd,name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) response message
    */
    proc deleteMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        if v {try! writeln("%s %s".format(cmd,name));try! stdout.flush();}
        // delete entry from symbol table
        st.deleteEntry(name);
        return try! "deleted %s".format(name);
    }

    /* 
    Takes the name of data referenced in a msg and searches for the name in the provided sym table.
    Returns a string of info for the sym entry that is mapped to the provided name.

    :arg reqMsg: request containing (cmd,name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
     */
    proc infoMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        if v {try! writeln("%s %s".format(cmd,name));try! stdout.flush();}
        // if name == "__AllSymbols__" passes back info on all symbols
        return st.info(name);
    }
    
    /* 
    query server configuration...
    
    :arg reqMsg: request containing (cmd)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
     */
    proc getconfigMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        if v {try! writeln("%s".format(cmd));try! stdout.flush();}
        return getConfig();
    }

    /* 
    query server total symbol table data memory
    
    :arg reqMsg: request containing (cmd)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
     */
    proc getmemusedMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        if v {try! writeln("%s".format(cmd));try! stdout.flush();}
        return st.memUsed():string;
    }
    
    /* 
    Response to __str__ method in python str convert array data to string 

    :arg reqMsg: request containing (cmd,name)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
   */
    proc strMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var printThresh = try! fields[3]:int;
        if v {try! writeln("%s %s %i".format(cmd,name,printThresh));try! stdout.flush();}
        return st.datastr(name,printThresh);
    }

    /* Response to __repr__ method in python.
       Repr convert array data to string 
       
       :arg reqMsg: request containing (cmd,name)
       :type reqMsg: string 

       :arg st: SymTab to act on
       :type st: borrowed SymTab 

       :returns: (string)
      */ 
    proc reprMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var printThresh = try! fields[3]:int;
        if v {try! writeln("%s %s %i".format(cmd,name,printThresh));try! stdout.flush();}
        return st.datarepr(name,printThresh);
    }


    /*
    Creates a sym entry with distributed array adhering to the Msg parameters (start, stop, stride)

    :arg reqMsg: request containing (cmd,start,stop,stride)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
    */
    proc arangeMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var start = try! fields[2]:int;
        var stop = try! fields[3]:int;
        var stride = try! fields[4]:int;
        // compute length
        var len = (stop - start + stride - 1) / stride;
        // get next symbol name
        var rname = st.nextName();
        if v {try! writeln("%s %i %i %i : %i , %s".format(cmd, start, stop, stride, len, rname));try! stdout.flush();}
        
        var t1 = Time.getCurrentTime();
        var aD = makeDistDom(len);
        var a = makeDistArray(len, int);
        writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        t1 = Time.getCurrentTime();
        forall i in aD {
            a[i] = start + (i * stride);
        }
        writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        st.addEntry(rname, new shared SymEntry(a));
        return try! "created " + st.attrib(rname);
    }            

    /* 
    Creates a sym entry with distributed array adhering to the Msg parameters (start, stop, len)

    :arg reqMsg: request containing (cmd,start,stop,len)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
    */
    proc linspaceMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var start = try! fields[2]:real;
        var stop = try! fields[3]:real;
        var len = try! fields[4]:int;
        // compute stride
        var stride = (stop - start) / (len-1);
        // get next symbol name
        var rname = st.nextName();
        if v {try! writeln("%s %r %r %i : %r , %s".format(cmd, start, stop, len, stride, rname));try! stdout.flush();}

        var t1 = Time.getCurrentTime();
        var aD = makeDistDom(len);
        var a = makeDistArray(len, real);
        writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        t1 = Time.getCurrentTime();
        forall i in aD {
            a[i] = start + (i * stride);
        }
        a[0] = start;
        a[len-1] = stop;
        writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

        st.addEntry(rname, new shared SymEntry(a));
        return try! "created " + st.attrib(rname);
    }

    /* 
    Sets all elements in array to a value (broadcast) 

    :arg reqMsg: request containing (cmd,name,dtype,value)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
    */
    proc setMsg(reqMsg: string, st: borrowed SymTab): string {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var dtype = str2dtype(fields[3]);
        var value = fields[4];

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError(pn,name);}

        select (gEnt.dtype, dtype) {
            when (DType.Int64, DType.Int64) {
                var e = toSymEntry(gEnt,int);
                var val: int = try! value:int;
                if v {try! writeln("%s %s to %t".format(cmd,name,val));try! stdout.flush();}
                e.a = val;
                repMsg = try! "set %s to %t".format(name, val);
            }
            when (DType.Int64, DType.Float64) {
                var e = toSymEntry(gEnt,int);
                var val: real = try! value:real;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:int));try! stdout.flush();}
                e.a = val:int;
                repMsg = try! "set %s to %t".format(name, val:int);
            }
            when (DType.Int64, DType.Bool) {
                var e = toSymEntry(gEnt,int);
                value = value.replace("True","true");
                value = value.replace("False","false");
                var val: bool = try! value:bool;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:int));try! stdout.flush();}
                e.a = val:int;
                repMsg = try! "set %s to %t".format(name, val:int);
            }
            when (DType.Float64, DType.Int64) {
                var e = toSymEntry(gEnt,real);
                var val: int = try! value:int;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:real));try! stdout.flush();}
                e.a = val:real;
                repMsg = try! "set %s to %t".format(name, val:real);
            }
            when (DType.Float64, DType.Float64) {
                var e = toSymEntry(gEnt,real);
                var val: real = try! value:real;
                if v {try! writeln("%s %s to %t".format(cmd,name,val));try! stdout.flush();}
                e.a = val;
                repMsg = try! "set %s to %t".format(name, val);
            }
            when (DType.Float64, DType.Bool) {
                var e = toSymEntry(gEnt,real);
                value = value.replace("True","true");
                value = value.replace("False","false");                
                var val: bool = try! value:bool;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:real));try! stdout.flush();}
                e.a = val:real;
                repMsg = try! "set %s to %t".format(name, val:real);
            }
            when (DType.Bool, DType.Int64) {
                var e = toSymEntry(gEnt,bool);
                var val: int = try! value:int;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:bool));try! stdout.flush();}
                e.a = val:bool;
                repMsg = try! "set %s to %t".format(name, val:bool);
            }
            when (DType.Bool, DType.Float64) {
                var e = toSymEntry(gEnt,int);
                var val: real = try! value:real;
                if v {try! writeln("%s %s to %t".format(cmd,name,val:bool));try! stdout.flush();}
                e.a = val:bool;
                repMsg = try! "set %s to %t".format(name, val:bool);
            }
            when (DType.Bool, DType.Bool) {
                var e = toSymEntry(gEnt,bool);
                value = value.replace("True","true");
                value = value.replace("False","false");
                var val: bool = try! value:bool;
                if v {try! writeln("%s %s to %t".format(cmd,name,val));try! stdout.flush();}
                e.a = val;
                repMsg = try! "set %s to %t".format(name, val);
            }
            otherwise {return unrecognizedTypeError(pn,fields[3]);}
        }
        return repMsg;
    }
    

}
