#######################
Quickstart
#######################

This guide assumes you have satisfied the :ref:`prerequisites-label` and followed the :ref:`installation-label` to build the arkouda server. Also, both your ``PATH`` and ``PYTHONPATH`` environment variables should contain the arkouda root directory.

**********************
Launch Arkouda Server
**********************

In a terminal, run the arkouda server program with one locale

.. code-block:: none

  $ arkouda_server -nl 1

You should see a startup message like

.. code-block:: none

  arkouda server version = 0.0.9-2019-09-23
  ...
  server listening on node01:5555

The last line is the most important, because it contains the hostname and port required for the client to connect to the server.

******************************
Connect the Python 3 Client
******************************

In another terminal window, launch an interactive Python 3 session, such as ``ipython`` or ``jupyter notebook`` (both included with the Anaconda distribution). To connect to the arkouda server, you must import the arkouda module and call connect with the hostname and port from the server startup message. In Python, run

.. code-block:: python

   >>> import arkouda as ak
   >>> ak.connect('node01', 5555)
   ...
   connected to tcp://node01:5555

substituting the hostname and port appropriately (defaults are 'localhost' and 5555).

******************************
Simple Computations
******************************

Create and sum an array
=========================

The following code creates an arkouda ``pdarray`` that resides on the arkouda server and performs a server-side computation, returning the result to the Python client.

.. code-block:: python

   # Create a server-side array with integers from 1 to N inclusive
   # This syntax is from NumPy
   >>> N = 10**6
   >>> A = ak.arange(1, N+1, 1)
   # Sum the array, returning the result to Python
   >>> print(A.sum())
   # Check the result
   >>> assert A.sum() == (N * (N + 1)) // 2

Array arithmetic
=========================
   
Now, we will perform an operation on two ``pdarray`` objects to create a new ``pdarray``. This time, the result will not be returned to the Python client, but will be stored on the server. In general, only scalar results are automatically returned to Python; ``pdarray`` results remain on the server unless explicitly transferred by the user (see :py:meth:`arkouda.pdarray.to_ndarray`).

.. code-block:: python

   # Generate two (server-side) arrays of random integers 0-9
   >>> B = ak.randint(0, 10, N)
   >>> C = ak.randint(0, 10, N)
   # Multiply them (server-side)
   >>> D = B * C
   # Print a small representation of the array
   # This does NOT move the array to the client
   >>> print(D)
   # Get the min and max values
   # Because these are scalars, they live in Python
   >>> minVal = D.min()
   >>> maxVal = D.max()
   >>> print(minVal, maxVal)

Indexing
=========================

Arkouda ``pdarray`` objects support most of the same indexing and assignment syntax of 1-dimensional NumPy ``ndarray``s (arkouda currently only supports 1-D arrays). This code shows two ways to get the even elements of ``A`` from above: with a slice, and with logical indexing.

.. code-block:: python

   # Use a slice
   >>> evens1 = A[1::2]
   # Create a logical index
   # Bool pdarray of same size as A
   >>> evenInds = ((A % 2) == 0)
   # Use it to get the evens
   >>> evens2 = A[evenInds]
   # Compare the two (server-side) arrays
   >>> assert (evens1 == evens2).all()

Sorting
===========================
   
Sorting arrays is a ubiquitous operation, and it is often useful to use the sorting of one array to order other arrays. Like NumPy, arkouda provides this functionality via the ``argsort`` function, which returns a permutation vector that can be used as an index to order other arrays. Here, we will order the arrays ``B`` and ``C`` from above according to the product of their elements (``D``).

.. code-block:: python

   # Compute the permutation that sorts the product array
   >>> perm = ak.argsort(D)
   # Reorder B, C, and D
   >>> B = B[perm]
   >>> C = C[perm]
   >>> D = D[perm]
   # Check that D is monotonically non-decreasing
   >>> assert (D[:-1] <= D[1:]).all()
   # Check that reordered B and C still produce D
   >>> assert ((B * C) == D).all()

And More
=====================

See the :ref:`usage-label` section for the full list of operations supported on arkouda arrays. These operations are quite composable and can be used to implement more complex algorithms as in the :ref:`examples-label` section.

******************************
Shutdown the server (optional)
******************************

If desired, you can shutdown the arkouda server from a connected client with

.. code-block:: python

   >>> ak.shutdown()

This command will delete all server-side arrays and cause the ``arkouda_server`` process in the first terminal to exit.
