### Changes in arkouda-0.0.6
------------------------
 * added toys/ subdir with some prototype sorting code
 * 


### Changes in arkouda-0.0.5
------------------------
 * finally got a BlockDist to work properly -- thanks Brad ;-)
 * changed ak.startup() to ak.connect() also added ak.disconnect()
 * added initial version of value_counts
 * added a[pdarray]=value indexing
 * changed to bool_ = type(True) in arkouda.py
 * added a[pdarray]=pdarray indexing
 * compiled with chapel 1.19 and cleaned up compilation warnings
 * substituted domain.contains() for domain.member()
 * fixed up parallel scans on bool arrays with int cast through a copy to int array (blah!)
 * created github repo
 * added backward compat code for 1.18
 * added some prototype python arkouda/numpy check/test code


### Changes in arkouda-0.0.4
------------------------
 * added threshold to pdarray.__iter__() to limit comms with arkouda_server
 * refactored how dist domain mapped arrays for symbol table entries
 * added some more tests to ak_test.py
 * simplified ones_like and zeros_like in arkouda.py
 * added bitwise {&,|,>>,<<,^,~} operators to int64
 * added unary-
 * added initial version of histogram
 * added bounds checking on pdarray indexing "a[iv]" where iv is an int64 pdarray
 * moved all operator stuff into OperatorMgs.chpl
 * factored the code put config const/param into ServerConfig.chpl module
 * added initital version of in1d
 * added initial version of randint
 * cleaned up ValueError vs. TypeError in arkouda.py
 * indexing operations moved to IndexingMsg.chpl
 * added initial version of unique


### Changes in arkouda-0.0.3
------------------------
 * changed over to distributed domain mapped arrays for arkouda server
 * added stdout.flush() after verbose messages to overcome buffering on big runs
 * added timer to report processing time around main loop
 * a little optimization of arange and linspace


### CHANGES in arkouda-0.0.2
------------------------
 * fixed bug in linspace in MsgProcessing.chpl
 * changed to using DType in all GenSymOps.chpl module from dynamic casts
 * simplified messaging in hpda.py
 * refactored binary op overloading in pdarray class
 * overloaded pdarray __str__ and __repr__, server messages and support also
 * added server error prop to raise exception in python 
 * refactored errors in MsgProcessing.chpl
 * changed name from hpda to arkouda (Greek for Bear)
 * fixed "/ and /= aka __truediv__" to return float64 and added "// and //= aka __floordiv__"
 * fixed ones and zeros to behave like numpy
 * added slices to __getitem__ and it does negative strides now!
 * added bool dtype
 * added relops
 * server gives out names instead of python
 * added sum, prod, any, and all reductions
 * added elemental functions log, exp, and abs
 * add numpy scans like cumsum and cumprod
 * add pdarray indexing b[] = a[iv[]],
   iv[] can be either int64 which does gather or bool which does a compress-out-false-indicies


### ISSUES
------
 * iv:pdarray:int64 = ak.argsort(pdarray)
    returns an index vector which sorts the original pdarray
    currently only needed for int64
    CountingSort.chpl has a start at this

 * 1D Scans in 1.19.0 of bool array cast to int need a copy made to parallelize

 * add uint64 or decide to use uint64 or int64 only

 * add some more random pdarray generators
 * add array() to turn a list into an pdarray
 * add nd2pdarray to turn an ndarray into a pdarray
 * add simple I/O array-to-file and array-from-file
 * add more float and int binops
 * add more float and int opeq
 * add argmax(maxloc) and argmin(minloc) reductions
 * fix sorta broken implementation of bools (Python3 vs. Numpy behavior)
    and bool is a type not a string like I have it (~ vs. not)
 * fix arange to behave like numpy
