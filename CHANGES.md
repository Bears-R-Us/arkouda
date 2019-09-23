### Changes in arkouda-0.0.9
------------------------
 * more optimized groupby/aggregate using segmented scans
 * fixed in1d bug
 * added in1d unit test
 * added groupby unit test
 * refactored a bunch of code
 * doubled speed of binary operators
 * much faster sort added
 * multi-level groupby added
 * many other changes lost track of them all

### Changes in arkouda-0.0.8
------------------------
 * added initial ak.argsort functionality
 * added initial ak.is_sorted functionality
 * added unit test to tests directory for ak.argsort
 * optimized argsort
 * added ak.where
 * opt some a[pdarray] gather-indexing
 * added initial ak.GroupBy and ak.aggregate functionality
 * added optimized per-locale version of groupby and aggregate

### Changes in arkouda-0.0.7
------------------------
 * fixed multi-file/multi-threaded bug in HDF5 I/O
 * improved HDF5 I/O error handling + glob filenames
 * added initial versions of ak.array() and pdarray.to_ndarray()
 * refactored histogram and some other procedures using helper nested procedures
 * added more operators for bool
 * added min and max reductions
 * optimized ak.int1d()

### Changes in arkouda-0.0.6
------------------------
 * added toys/ subdir with some prototype sorting code
 * made some changes to make naming consistent in Python and Chapel code
 * added initial HDF5 I/O
 * added a[slice] = value indexing
 * added a[slice] = pdarray indexing
 * added initial ak.argmin/ak.argmax functionality
 * added some different optimizations and optional return of the counts from ak.unique
 * added more error handling to GenSymIO (HDF5 I/O)

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

