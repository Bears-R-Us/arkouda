
# Adding Your First Feature


This guide describes how to add new functionality to arkouda. 
For each new feature, you need to add the client-side interface in Python and the server-side implementation in Chapel.
To demonstrate this process, this guide walks through the process of implementing a custom function ``times2`` that will multiply all elements of an array by 2.

## Adding Python Functionality (Client Interface)


Python functions should follow the API of NumPy or Pandas, were possible. In general, functions should conform to the following:

1. Be defined somewhere in the `arkouda` subdirectory, such as in ``arkouda/pdarraysetops.py``
2. Have a complete docstring in (`NumPy` format)[https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard]
3. Check argument types and properties, raising exceptions if necessary
4. Send a request message using `generic_msg(request)`
5. Process the reply message
6. Possibly create one or more `pdarray` objects
7. Return any results

### Example

First, add the client-side implementation to construct the message to the server:

```
@typechecked
def times2(pda: pdarray) -> pdarray:
    """
    Returns a pdarray with each entry double that of the input.

    Parameters
    ----------
    pda : pdarray
        The array to double.

    Returns
    -------
    pdarray
        The doubled array
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg(cmd="times2", args={"arg1" : pda})
        return create_pdarray(repMsg)
    else:
        raise TypeError("times2 only supports pdarrays.")
```
Note that the function signature makes use of Python's optional type checking features. 
You can learn more about Python type checking in (this documentation)[https://docs.python.org/3/library/typing.html].

Second, add your function to the `__all__` list near the top of the file.
```
__all__ = ["in1d", "concatenate", "union1d", "intersect1d", "setdiff1d", "setxor1d", "times2"]
```


## Adding Functionality to the Arkouda Server


Your contribution must include all the machinery to process a command from the client, in addition to the logic of the computation. 
This is broken into function(s) that implement the actual operation, a function that processes the command message and calls the appropriate implementation, and code to register the message processing function with the message dispatch system.
For the `times2` example, we will add our functions to the `ArraySetOps` and `ArraySetOpsMsg` modules for the sake of simplicity.

When the client issues a command `times2 arg1` to the arkouda server, this is what typically happens:

1. The `select` block in `ServerDaemon.chpl` sees "times2", looks it up in the `commandMap`, and calls the function responsible for processing the message: `times2Msg`.

1. The `times2Msg` function is found via the `ArraySetopsMsg` module, which contains `use ArraySetops` and thus gets all symbols from the `ArraySetops` module where the implementation function `times2()` is defined.

1. The `times2Msg()` function (in the `ArraySetopsMsg` module) parses and executes the command by

   1. Casting any scalar args

   1. Looking up `pdarray` (`GenSymEntry`) args in the symbol table with `getGenericTypeArrayEntry`

   1. Checking dtypes of arrays and branching to corresponding code

   1. Casting `GenSymEntry` objects to correct types with `toSymEntry()`

   1. Executing the operation, usually on the array data `entry.a`

   1. If necessary, creating new `SymEntry` and adding it to the symbol table with `st.addEntry()`

   1. Returning an appropriate message string

      1. If the return is an array, "created <attributes>"

      1. If the return is multiple arrays, one creation string per array, joined by "+"

      1. If the return is a scalar, "<dtype> <value>"

      1. If any error occurred, then "Error: <message>" (see `ServerErrorStrings.chpl` for functions to generate common error strings)

Example
-------

First, define your message processing logic in `src/ArraySetopsMsg.chpl` in the following manner:

```
module ArraySetopsMsg {

    ...
    /* 
    Parse, execute, and respond to a times2 message 
    :arg cmd: request command
    :type reqMsg: string 
    :arg msgArgs: request arguments
    :type msgArgs: borrowed MessageArgs
    :arg st: SymTab to act on
    :type st: borrowed SymTab 
    :returns: (MsgTuple) response message
    */
    proc times2Msg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message
        
        var vName = st.nextName(); // symbol table key for resulting array

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("arg1"), st);

        select gEnt.dtype {
            when DType.Int64 {
                var e = toSymEntry(gEnt,int);

                var aV = times2(e.a);
                st.addEntry(vName, createSymEntry(aV));

                repMsg = "created " + st.attrib(vName);
                asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // add additional when blocks for different data types...
            otherwise {
                var errorMsg = notImplementedError("times2",gEnt.dtype);
                asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);           
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }

    ...
}
```

Second, define your operation implementation in `src/ArraySetops.chpl` in the following manner:

```
module ArraySetops {
    
    ...

    // returns input array, doubled.
    proc times2(a: [] ?t) throws {
        var ret = a * 2; //scalar promotion
        return ret;
    }

    ...
}
```

Finally, register your new function within the commandMap back in `src/ArraySetopsMsg.chpl` in the following manner:


```
module ArraySetopsMsg {
    ...

    use CommandMap;
    ...
    resisterFunction("times2", times2Msg, getModuleName());
}
```

Now, you should be able to rebuild and launch the server and use your new feature. We close with a client-side python script that uses the new feature.

```
import arkouda as ak
ak.connect()
undoubled = ak.arange(0,10)
doubled = ak.times2(undoubled)
```