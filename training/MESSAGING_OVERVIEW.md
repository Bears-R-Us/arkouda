# Communication Between Client and Server
This document presents a high-level overview of Arkouda's Client and Server interaction. By the end of this document you will understand the process of:
- Configuring client code to format and send a message/request to the server
- Configuring a Chapel module for the Arkouda Server code 
    - Receive and parse the request message
    - Execute the request
    - Sending a response message to the client
- Receiving the response message on the client.

<a id="toc"></a>
## Table of Contents
1. [Overview](#msgOverview)
2. [Sending a Request to the Server](#sendReq)
3. [Receiving a Request on the Server](#receiveReq)
4. [Creating a new Chapel Module](#newModule)
5. [Try-It Example](#tryIt)

<a id="msgOverview"></a>
## Overview

Arkouda uses ZeroMQ to handle messaging between our Python client and Chapel Server. This communication enables Arkouda to process large scale data efficiently by leveraging Chapel's compiled and parallel nature. 

Message arguments to be sent to the server must be in a valid JSON format, preferably a python dictionary. These arguments are sent to the server along with a `cmd` string value which lets the server know which function to use to process the arguments. To accomplish this, we use a command map structure in Chapel to map a `cmd` string name to a Chapel function within a server Module. Processing functions are linked to command strings using a register function. These functions return a Response Message in a string format. We recommend using a JSON string as the Response Message whenever possible.

At the root level there is a `ServerModules.cfg` file that lists all the active Chapel server modules available for Command Map arguments. To add a new module, simply add the name of the module to this file. To learn how to structure a new Chapel Module, go to [Creating a New Module](#newModule).

<a id="sendReq"></a>
## Sending a Request to the Server

In general, a server request will look something like this:

`repMsg = cast(str, generic_msg(cmd="serverFunction", args={"arg1": arg1, "arg2": arg2}))`

To break this request down, `cmd` is the string name that maps to the function you want to use in Chapel. In most cases the string name will be almost identical to the Chapel function name to keep things easy to maintain. `args` is a JSON of all the arguments being passed to the server for processing.

We cast the return from `generic_msg` to a `str` in most cases to ensure our return is in a string format. The alternative is `memoryview`, which is a binary representation that is mostly used for large data transfer methods such as `array()` and `to_ndarray()`.

It is important to note that the Arkouda client does some data enrichment to the dictionary passed in the `args` field. The JSON string that is sent to the server for the dictionary `{"arg1": "First Arg", "arg2": 2}` would be:

```python
(2,
 '["{\\"key\\": \\"arg1\\", \\"objType\\": \\"VALUE\\", \\"dtype\\": \\"str\\", \\"val\\": \\"First Arg\\"}", \
 "{\\"key\\": \\"arg2\\", \\"objType\\": \\"VALUE\\", \\"dtype\\": \\"int\\", \\"val\\": \\"2\\"}"]')
```

The `key` field is how you access your values through the provided name. `objType` is used to let the server know what kind of value is being passed in this field. Possible `objType` values are: "PDARRAY", "SEGSTRING", "LIST", "DICT", "VALUE", "DATETIME", and "TIMEDELTA". `dtype` represents the datatype of the value in this field. And finally, `val` is the actual value of the field.


<a id="receiveReq"></a>
## Receiving a Request on the Server

As mentioned above, the Command Map handles mapping the given `cmd` string to the proper function in the Chapel based on the available Chapel Modules. In order for a function within a module to be found by the Command Map it must first be registered. An example of this, using the request in the previous section as an example, would be as follows:

`registerFunction("serverFunction", serverFunctionMsg, getModuleName());`

The three components of the `registerFunction` method are the `cmd` string, the Chapel function name, and the Module name which is supplied by `getModuleName()`.

It is important the Chapel function is formatted correctly or the server will not compile. The expected formatting for a `serverFunctionMsg` is:
```chapel
proc serverFunctionMsg(cmd: string, payload: string, argSize: int, st: borrowed SymTab): MsgTuple throws {}
```

The parameters in this function each server a specific purpose:
- `cmd` is the command string used to access this function. It is used mostly for debugging as the `cmd` string used to access the function will be captured in the logger messages.
- `payload` is the JSON string of arguments passed from the client.
- `argSize` lets Chapel know how many arguments are in the payload JSON so it can properly create a map using the JSON.
- `st: borrowed SymTab` is a reference parameter to the Server's Symbol Table. The Symbol Table acts like Arkouda's database; storing all information related to a data structure that can then be referenced from the Client without having to pass large amounts of data back and forth.

All messages sent to the server are in JSON format. Before any arguments sent from the client can be used, they must be parsed. To do this use the function `parseMessageArgs(payload, argSize)`. This will parse your message arguments into a Map that can then be accessed using accessor methods. The most common of these accessor methods is `getValueOf(argName)` which will return the value of an argument that matches the provided name.

For example, assuming the server request in the above section is what is being received, there are two arguments in the JSON, `arg1` and `arg2`. To access these arguments in Chapel, we will parse the message, or `payload`, into a variable called `msgArgs`. Then using the `getValueOf()` accessor method, we will set variables `arg1` and `arg2` equal to the values of their corresponding JSON arguments.

```chapel
var msgArgs = parseMessageArgs(payload, argSize);
var arg1 = msgArgs.getValueOf("arg1"); // str dtype example
var arg2 = msgArgs.getIntValue("arg2"); // int dtype example
```

Now the values of `arg1` and `arg2` are available for use within this Chapel function. Note that the function used can differ depending on the dtype of the expected value. `msgArgs.getValueOf()` is the base case that will return your value as a string. But there are other accessor methods that will cast the return automatically to various types:
```chapel
msgArgs.getIntValue()  // int64
msgArgs.getUIntValue()  // uint64
msgArgs.getUInt8Value()  // uint8
msgArgs.getRealValue()  // real
msgArgs.getBoolValue()  // bool
msgArgs.getValueAsType(type t)  // returns as provided type
msgArgs.getList(size s)  // List of int values of provided size
```

<a id="newModule"></a>
## Creating a New Chapel Module

Adding a new Chapel module is simple. Create a file of the same name as your Chapel module. If there is no communication between Client and Server in your module, there's nothing special needed! Otherwise, there are two important things to remember when creating a new module that have communication between Client and Server:

- Add the module name to `serverModules.cfg`. (This file sits at the root level of the Arkouda repo).
- At the end of your Chapel module, include the following with a `registerFunction` method call for each function in your module that will receive a request from the Client.
```chapel
   use CommandMap;
   registerFunction(stringName, functionName, moduleName);
   ...
   registerFunction(stringNameN, functionNameN, moduleName);
```

Putting everything together, the client's request to the server will look something like this:

```python
from typing import cast
from arkouda.client import generic_msg

def example(arg="First Argument"):
  arg1 = arg
  arg2 = 10
  repMsg = cast(str, generic_msg(cmd="serverFunction", args={"arg1": arg1, "arg2": arg2}))
  left, right = repMsg.split("+")
  return left, right
```

<a id="exMod"></a>
Using this request, this is an example of a Chapel module that will process and return a string based on the given arguments `arg1` and `arg2`:
```chapel
module ExampleModule {

    use ServerConfig;
    
    use Reflection;
    use ServerErrors;
    use Message;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    
    proc exampleFunction(cmd: string, payload: string, argSize: int, 
                                    st: borrowed SymTab): MsgTuple throws {
        var repMsg: string; // response message

        var msgArgs = parseMessageArgs(payload, argSize);
        var arg1 = msgArgs.getValueOf("arg1");
        var arg2 = msgArgs.getIntValue("arg2");
        
        repMsg = "arg1: %s + arg2: %s".format(arg1, arg2);
        
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    
    use CommandMap;
    registerFunction("serverFunction", exampleFunction, getModuleName());
}
```

The `serverModules.cfg` must have `ExampleModule` added to the bottom of the module list:
```shell
# Comment out modules using a `#` to exclude
# a module from a build

ArraySetopsMsg
KExtremeMsg
ArgSortMsg
SegmentedMsg
DataFrameIndexingMsg
OperatorMsg
RandMsg
IndexingMsg
UniqueMsg
In1dMsg
HistogramMsg
SequenceMsg
SortMsg
ReductionMsg
EfuncMsg
ConcatenateMsg
JoinEqWithDTMsg
RegistrationMsg
CastMsg
BroadcastMsg
FlattenMsg
StatsMsg
TimeClassMsg
HDF5Msg
ParquetMsg
HDF5MultiDim
ExampleModule
```

If we were to print the value of `repMsg` on the client after performing this request, the output would be: `'arg1: First Argument + arg2: 10'`

The `return` from the client function would be:
`"First Argument", 10`

<a id="tryIt"></a>
# Try It!

Now it's your turn to use the information presented in this document to create your own module, complete with a request to the Server and response back to the Client!

1. In the root-level directory `arkouda`, create a new python file `mymodule.py`
   - At a minimum for server communication, you will need to import:
     - `from typing import cast`
     - `from arkouda.client import generic_msg`
2. Create a new function which takes in as many arguments as you want
   - `def my_request(arg1, arg2, ...):`
3. Send your arguments to the Server
   - `repMsg = cast(str, generic_msg(cmd="myServerFunction", args={"arg1": arg1, "arg2": arg2, ...}))`
4. Print your response from the server
   - `print(repMsg)`
5. Using the [example module](#exMod) above as a reference, create a new Chapel module within the root-level `src` directory
    - This module should include:
      - A `CommandMap` registered function that your request is being sent into
      - JSON parsing of your message and some form of manipulation of your arguments. Ex. addition, subtraction, string-reversal, string-concatenation
      - The creation of a `repMsg` string
      - The return of a `new MsgTuple` of `MsgType.NORMAL`
    - **Remember to add your Chapel file name to `ServerModules.cfg`**
6. To test your new module, you'll first have to use the terminal to navigate to the root-level directory of Arkouda then run `make` to compile your new Chapel code.
7. Once Chapel compiles, you can then launch the Arkouda server using `./arkouda_server -nl 1`
8. Using `ipython` you can then easily test your new module
```python
import arkouda as ak
from arkouda.mymodule import my_request

ak.connect() # Connects to the server running locally

my_request("First Argument", 10, ...)
```