
# Adding Your First Feature


This guide describes how to add new functionality to arkouda. 
For each new feature, you need to add the client-side interface in Python and the server-side implementation in Chapel.
To demonstrate this process, this guide walks through the process of implementing a custom function ``times2`` that will multiply all elements of an array by 2.

## Adding Python Functionality (Client Interface)


Python functions should follow the API of NumPy or Pandas, where possible. In general, functions should conform to the following:

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
    from arkouda.client import generic_msg
    if isinstance(pda, pdarray):
        rep_msg = generic_msg(cmd=f"times2<{pda.dtype},{pda.ndim}>", args={"arg1" : pda})
        return create_pdarray(rep_msg)
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

we will add a procedure called `times2` to the`ArraySetOpsMsg` module that will execute our command.

When the client issues a command like `times2<int64,1> arg1` to the arkouda server, this is what typically happens:

1. The `select` block in `ServerDaemon.chpl` sees the command "times2<int64,1>", looks it up in the `commandMap`, and calls the function responsible for processing the message.
    - This will call an instantiation of our `times2` procedure for 1D arrays with element type `int(64)`.

1. The `times2()` function (in the `ArraySetopsMsg` module) applies our operation to the array, returning a new array

1. The server generates a response to the client with a handle to the new array and its meta-data

Example
-------

Define a function in `src/ArraySetopsMsg.chpl` that takes an array argument and
multiplies it by `2` using a promoted operation:

```chapel
module ArraySetopsMsg {
    ...

    @arkouda.registerCommand
    proc times2(const ref arg1: [?d] ?t): [d] t {
        var ret = a * 2;
        return ret;
    }

    ...
}
```

Notice that the function is annotated with `@arkouda.registerCommand`. This
will tell Arkouda's build system to generate and register several commands in
the server's CommandMap. One command will be created for each combination of
array rank and dtype specified in the `registration-config.json` file. For
example, if the file has the following settings for `"array"`:

```json
"array": {
    "nd": [1,2],
    "dtype": ["int","real"]
}
```

then, these commands will be generated:
* "times2<int64,1>"
* "times2<int64,2>"
* "times2<float64,1>"
* "times2<float64,2>"

Note that the configuration file specifies Chapel type names, while the command
names contain their Python/Numpy counterparts.

Occasionally, the `registerCommand` annotation doesn't provide fine enough
control over a command's behavior. To gain lower-level access to Arkouda's
internal infrastructure the `instantiateAndRegister` annotation is also
provided. For an example of why it may be needed, look at `castArray` in
"castMsg.chpl" where the commands need to have two type arguments (one for the
array being cast, and another for the type it's cast to).

If we wanted to use `instantiateAndRegister` for `times2`, we could write:

```chapel
module ArraySetopsMsg {
    ...

    @arkouda.instantiateAndRegister
    proc times2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
        const x = st[msgArgs['arg1']]: borrowed SymEntry(array_dtype, array_nd);
        const y = new SymEntry(x.a * 2);
        return st.insert(y);
    }

    ...
}
```

With this annotation, the first three arguments must always have the same
names and types as above, and the return type must be `MsgTuple`. This
procedure manually pull's the array's name from the JSON arguments provided to
the command: `msgArgs['arg1']`. It then acquires the array symbol (a class that
wraps a Chapel array from the server's symbol table), and casts it to a
`SymEntry` with our array's particular type and rank. We then create a new
SymEntry with the result of our computation, and store its value in a new
variable `y`. Finally, a response for the server is generated and returned by
adding the new symbol to the symbol table with the `insert` method.

Using a command written with either of the above annotations, you should be
able to rebuild and launch the server and use your new feature. We close with a
client-side python script that uses the new feature:

```
import arkouda as ak
ak.connect()
undoubled = ak.arange(0,10)
doubled = ak.times2(undoubled)
```
