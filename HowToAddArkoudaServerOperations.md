# HowTo add Arkouda server operations

## Add functionality to `arkouda.py`

 * add function definition to `arkouda.py`
  * define function
  * check types, handle errors, and raise exceptions
  * send request message
  * process reply message
  * possibly create resulting pdarray object
  * return result

### Here is a simple example of the Python3 
```python3
def foo(pda):
    """
    Return the foo() of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("foo {}".format(pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))
```

## Add functionality to `arkouda_server.chpl`

 * edit `arkouda_server.chpl` in the `arkouda/src/` directory and add a when clause in the top-level `select cmd`
 * edit `MsgProcessing.chpl` and add a `use FooMsg;` to the top of the `MsgProcessing` module
 * edit/create `FooMsg.chpl` and put the `FooMsg` module in it
  * add `proc fooMsg()` to the module
  * add cmd string parsing to the procedure
  * check for errors and return `Error: ...` string if there is an error condition
  * lookup the the arguments of the command in the symbol table `st.lookup()` and check for existence
  * get name for result by calling `st.nextName()`
  * check dtype and branch to correct code to execute
  * cast generic symbol table entry to correct type
  * execute the operation
  * create and put the result in the symbol table
  * return a `"created <attributes>" string to the main loop

### Here is a simple example of adding to `src/arkouda_server.chpl`
```chapel
        // parse requests, execute requests, format responses
        select cmd
        {
	// ...
	    when "foo"             {repMsg = fooMsg(reqMsg, st);}
	// ...
	}

```

### Here is a simple example of adding a module to `src/MsgProcessing.chpl`
```chapel

module MsgProcessing
{
    use ServerConfig;

    use Time only;
    use Math only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use AryUtil;
    
    use OperatorMsg;
    // ...    
    use FooMsg;
    // ...
```

### here is a simple example of `src/FooMsg.chpl`
```chapel
module FooMsg
{
    // do foo on array a
    proc foo(a: [?aD] int): [aD] int {
        //...
        return(ret);
    }
    
    /* 
    Parse, execute, and respond to a foo message 

    :arg reqMsg: request containing (cmd,dtype,size)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (string) response message
    */
    proc FooMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];

        // get next symbol name
        var rname = st.nextName();
        
        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError("set",name);}

        // if verbose print action
        if v {try! writeln("%s %s: %s".format(cmd,name,rname)); try! stdout.flush();}

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
		var ret = foo(e);
		st.addEntry(rname, new shared SymEntry(ret));
            }
            otherwise {return unrecognizedTypeError("foo",gEnt.dtype);}
	}
        // response message
        return try! "created " + st.attrib(rname);
    }

}
```
