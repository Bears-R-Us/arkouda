
module MsgProcessing
{
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection only;
    use Memory;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use AryUtil;
    
    public use OperatorMsg;
    public use RandMsg;
    public use IndexingMsg;
    public use UniqueMsg;
    public use In1dMsg;
    public use HistogramMsg;
    public use ArgSortMsg;
    public use SortMsg;
    public use ReductionMsg;
    public use FindSegmentsMsg;
    public use EfuncMsg;
    public use ConcatenateMsg;
    public use SegmentedMsg;
    public use JoinEqWithDTMsg;
    public use RegistrationMsg;
    public use NewUnion1dMsg;
    
    /* 
    Parse, execute, and respond to a create message 

    :arg reqMsg: request containing (cmd,dtype,size)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) response message
    */
    proc createMsg(reqMsg: string, st: borrowed SymTab): string throws {
        var repMsg: string; // response message
        // split request into fields
        var (cmd, dtypestr, sizestr) = reqMsg.splitMsgToTuple(3);
        var dtype = str2dtype(dtypestr);
        var size = try! sizestr:int;

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
        // split request into fields
        var (cmd,name) = reqMsg.splitMsgToTuple(2);
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
        // split request into fields
        var (cmd,name) = reqMsg.splitMsgToTuple(2);
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
        var (cmd, _) = reqMsg.splitMsgToTuple(2); // split request into fields
        if v {try! writeln("%s".format(cmd));try! stdout.flush();}
        return getConfig();
    }

    /* 
    query server total memory allocated or symbol table data memory
    
    :arg reqMsg: request containing (cmd)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
     */
    proc getmemusedMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var (cmd, _) = reqMsg.splitMsgToTuple(2); // split request into fields
        if v {try! writeln("%s".format(cmd));try! stdout.flush();}
        if (memTrack) {
            return (memoryUsed():uint * numLocales:uint):string;
        }
        else {
            return st.memUsed():string;
        }
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
        // split request into fields
        var (cmd,name,ptstr) = reqMsg.splitMsgToTuple(3);
        var printThresh = try! ptstr:int;
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
        // split request into fields
        var (cmd,name,ptstr) = reqMsg.splitMsgToTuple(3);
        var printThresh = try! ptstr:int;
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
    proc arangeMsg(reqMsg: string, st: borrowed SymTab): string throws {
        var repMsg: string; // response message
//        var (cmd, start, stop, stride) = try! (reqMsg.splitMsgToTuple(4): (string, int, int, int));
        var (cmd, startstr, stopstr, stridestr) = reqMsg.splitMsgToTuple(4);
        var start = try! startstr:int;
        var stop = try! stopstr:int;
        var stride = try! stridestr:int;
        // compute length
        var len = (stop - start + stride - 1) / stride;
        // get next symbol name
        var rname = st.nextName();
        if v {try! writeln("%s %i %i %i : %i , %s".format(cmd, start, stop, stride, len, rname));try! stdout.flush();}
        
        var t1 = Time.getCurrentTime();
        var e = st.addEntry(rname, len, int);
        if v {writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}

        t1 = Time.getCurrentTime();
        ref ea = e.a;
        ref ead = e.aD;
        forall (ei, i) in zip(ea,ead) {
            ei = start + (i * stride);
        }
        if v {writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}

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
    proc linspaceMsg(reqMsg: string, st: borrowed SymTab): string throws {
        var repMsg: string; // response message
//        var (cmd, start, stop, len) = try! ( reqMsg.splitMsgToTuple(4): (string, real, real, int));
        var (cmd, startstr, stopstr, lenstr) = reqMsg.splitMsgToTuple(4);
        var start = try! startstr:real;
        var stop = try! stopstr:real;
        var len = try! lenstr:int;
        // compute stride
        var stride = (stop - start) / (len-1);
        // get next symbol name
        var rname = st.nextName();
        if v {try! writeln("%s %r %r %i : %r , %s".format(cmd, start, stop, len, stride, rname));try! stdout.flush();}

        var t1 = Time.getCurrentTime();
        var e = st.addEntry(rname, len, real);
        if v {writeln("alloc time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}

        t1 = Time.getCurrentTime();
        ref ea = e.a;
        ref ead = e.aD;
        forall (ei, i) in zip(ea,ead) {
            ei = start + (i * stride);
        }
        ea[0] = start;
        ea[len-1] = stop;
        if v {writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}

        return try! "created " + st.attrib(rname);
    }

    /* 
    Sets all elements in array to a value (broadcast) 

    :arg reqMsg: request containing (cmd,name,dtype,value)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string)
    :throws: `UndefinedSymbolError(name)`
    */
    proc setMsg(reqMsg: string, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var (cmd, name, dtypestr, value) = reqMsg.splitMsgToTuple(4);
        var dtype = str2dtype(dtypestr);

        var gEnt: borrowed GenSymEntry = st.lookup(name);

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
            otherwise {return unrecognizedTypeError(pn,dtypestr);}
        }
        return repMsg;
    }
}
