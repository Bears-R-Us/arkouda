******************
Adding a Benchmark
******************

Adding a benchmark can generally be completed by copying an existing benchmark and modifying only a few lines. Some steps to get started adding a benchmark:

1. Create a python file in ``arkouda/benchmarks`` named after the method you would like to add a benchmark for (e.g. argsort.py)
2. Copy the example code below, modified to call the method you would like to add a benchmark for instead of argsort
   
- note: The example code below will only work if your method is a standalone arkouda method. If instead your method is, for example, an attribute to the pdarray class, it would instead need to be written as ``perm = a.myMethod()``
   

Example
-------

.. code-block:: python
                
     import time, argparse
     import numpy as np
     import arkouda as ak

     TYPES = ('int64', 'float64')

     def time_ak_argsort(N_per_locale, trials, dtype):
         print(">>> arkouda argsort") #Name of method to be tested
         cfg = ak.get_config()
         N = N_per_locale * cfg["numLocales"]
         print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
         if dtype == 'int64':
             a = ak.randint(0, 2**32, N)
         elif dtype == 'float64':
             a = ak.randint(0, 1, N, dtype=ak.float64)

         timings = []
         for i in range(trials):
             start = time.time()
             perm = ak.argsort(a) #Your function goes here
             end = time.time()
             timings.append(end - start)
         tavg = sum(timings) / trials

         print("Average time = {:.4f} sec".format(tavg))
         bytes_per_sec = (a.size * a.itemsize) / tavg
         print("Average rate = {:.4f} GiB/sec".format(bytes_per_sec/2**30))

     def create_parser():
         parser = argparse.ArgumentParser(description="Measure performance of sorting an array of random values.")
         parser.add_argument('hostname', help='Hostname of arkouda server')
         parser.add_argument('port', type=int, help='Port of arkouda server')
         parser.add_argument('-n', '--size', type=int, default=10**8, help='Problem size: length of array to argsort')
         parser.add_argument('-t', '--trials', type=int, default=3, help='Number of times to run the benchmark')
         parser.add_argument('-d', '--dtype', default='int64', help='Dtype of array ({})'.format(', '.join(TYPES)))
         return parser

     if __name__ == "__main__":
         import sys
         parser = create_parser()
         args = parser.parse_args()
         if args.dtype not in TYPES:
             raise ValueError("Dtype must be {}, not {}".format('/'.join(TYPES), args.dtype))
         ak.verbose = False
         ak.connect(args.hostname, args.port)

         print("array size = {:,}".format(args.size))
         print("number of trials = ", args.trials)
         time_ak_argsort(args.size, args.trials, args.dtype)

         sys.exit(0)

Running your benchmark
======================

To ensure your benchmark is working, you can run it by running the command from your root directory:
``./benchmarks/run_benchmarks.py myMethod.py``
where "myMethod.py" is replaced with your filename.

Once everything is working here, correctness testing and numpy testing should be added to your benchmark in the following manner:

1. Add the code below to your benchmark file, modifying them in the same manor that you modified the Arkouda benchmark above

.. code-block:: python

     def time_np_argsort(N, trials, dtype):
         print(">>> numpy argsort")
         print("N = {:,}".format(N))
         if dtype == 'int64':
             a = np.random.randint(0, 2**32, N)
         elif dtype == 'float64':
             a = np.random.random(N)

         timings = []
         for i in range(trials):
             start = time.time()
             perm = np.argsort(a)
             end = time.time()
             timings.append(end - start)
         tavg = sum(timings) / trials

         print("Average time = {:.4f} sec".format(tavg))
         bytes_per_sec = (a.size * a.itemsize) / tavg
         print("Average rate = {:.4f} GiB/sec".format(bytes_per_sec/2**30))

     def check_correctness(dtype):
         N = 10**4
         if dtype == 'int64':
             a = ak.randint(0, 2**32, N)
         elif dtype == 'float64':
             a = ak.randint(0, 1, N, dtype=ak.float64)

         perm = ak.argsort(a)
         assert ak.is_sorted(a[perm])

2. Add these arguments to your ``create_parser()`` method

.. code-block:: python
   
    parser.add_argument('--numpy', default=False, action='store_true', help='Run the same operation in NumPy to compare performance.')
    parser.add_argument('--correctness-only', default=False, action='store_true', help='Only check correctness, not performance.')

3. Modify the lines after ``if __name__ ==  __main__:`` to include your added functionality

.. code-block:: python

    import sys
    parser = create_parser()
    args = parser.parse_args()
    if args.dtype not in TYPES:
        raise ValueError("Dtype must be {}, not {}".format('/'.join(TYPES), args.dtype))
    ak.verbose = False
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        for dtype in TYPES:
            check_correctness(dtype)
        sys.exit(0)
    
    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_argsort(args.size, args.trials, args.dtype)
    if args.numpy:
        time_np_argsort(args.size, args.trials, args.dtype)

4. Now try running your benchmark with the additional functionality:

numpy test: ``./benchmarks/run_benchmarks.py myMethod --numpy``
correctness test: ``./benchmarks/run_benchmarks.py myMethod --correctness-only``


Updating Graphs
===============

To get your benchmark to be tracked using graphs, you will need to:

1. Add a ``.perfkeys`` file in ``benchmarks/graph_infra`` following the convention of existing files
2. Update ``arkouda.graph`` to select your new perfkeys, just like the others

More information about the benchmark graphs available at ``benchmarks/graph_infra/README.md``
