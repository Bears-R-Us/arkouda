***************
Contributing
***************

This section describes how to add new functionality to arkouda.

Adding Python Functionality
===========================

Python functions should follow the API of NumPy or Pandas, were possible. In general, functions should conform to the following:

1. Be defined in ``arkouda.py``
2. Have a complete docstring in `NumPy format <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
3. Check argument types and properties, raising exceptions if necessary
4. Send a request message using ``generic_msg(request)``
5. Process the reply message
6. Possibly create one or more ``pdarray`` objects
7. Return any results

Example
-------

.. code-block:: python

    def foo(pda):
    """
    Return the foo() of the array.

    Parameters
    ----------
    pda : pdarray
        The array to foo

    Returns
    -------
    pdarray
        The foo'd array
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("foo {}".format(pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

Adding Functionality to the Arkouda Server
==========================================

Your contribution must include all the machinery to process a command from the client, in addition to the logic of the coputation. When the client issues a command ``foo arg1 arg2 ...`` to the arkouda server, this is what typically happens:

#. The ``select`` block in ``arkouda_server.chpl`` sees "foo" and calls ``fooMsg(reqMsg, st)``, passing the command string and the symbol table.

#. The ``fooMsg`` function is found via the ``MsgProcessing`` module, which contains ``use FooMsg`` and thus gets all symbols from the ``FooMsg`` module where ``fooMsg()`` is defined.

#. The ``fooMsg()`` function (in the ``FooMsg`` module) parses and executes the command by

   #. Splitting the command string

   #. Casting any scalar args

   #. Looking up ``pdarray`` (``GenSymEntry``) args in the symbol table with ``st.lookup(arg)`` and checking for ``nil`` result

   #. Checking dtypes of arrays and branching to corresponding code

   #. Casting ``GenSymEntry`` objects to correct types with ``toSymEntry()``

   #. Executing the operation, usually on the array data ``entry.a``

   #. If necessary, creating new ``SymEntry`` and adding it to the symbol table with ``st.addEntry()``

   #. Returning an appropriate message string

      #. If the return is an array, "created <attributes>"

      #. If the return is multiple arrays, one creation string per array, joined by "+"

      #. If the return is a scalar, "<dtype> <value>"

      #. If any error occurred, then "Error: <message>" (see ``ServerErrorStrings.chpl`` for functions to generate common error strings)

Example
-------

First, in ``src/arkouda_server.chpl``, add a ``when`` statement to register the "foo" command:

.. code-block:: chapel

   // parse requests, execute requests, format responses
   select cmd
   {
       // ...
       when "foo"             {repMsg = fooMsg(reqMsg, st);}
       // ...
   }

Next, in the ``MsgProcessing`` module, add ``use FooMsg;`` in the appropriate location:

.. code-block:: chapel

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

Then, define your argument parsing and function logic in ``src/FooMsg.chpl`` in the following manner:

.. code-block:: chapel

   module FooMsg
   {
       use MultiTypeSymEntry;
       use ServerErrorStrings;
       
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
