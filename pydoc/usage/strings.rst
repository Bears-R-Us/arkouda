*********************
Strings in Arkouda
*********************

Like NumPy, Arkouda supports arrays of strings, but whereas in NumPy arrays of strings are still ``ndarray`` objects, in Arkouda the array of strings is its own class: ``Strings``.

In order to efficiently store strings with a wide range of lengths, Arkouda uses a "segmented array" data structure, comprising:

* ``bytes``: A ``uint8`` array containing the concatenated bytes of all the strings, separated by null (0) bytes.
* ``offsets``: A ``int64`` array with the start index of each string

Performance
===========

Because strings are a variable-width data type, and because of the way Arkouda represents strings, operations on strings are considerably slower than operations on numeric data. Use numeric data whenever possible. For example, if your raw data contains string data that could be represented numerically, consider setting up a processing pipeline performs the conversion (and stores the result in HDF5 format) on ingest.

I/O
===========

Arrays of strings can be transferred between the Arkouda client and server using the ``arkouda.array`` and ``Strings.to_ndarray`` functions (see :ref:`IO-label`). The former converts a Python list or NumPy ``ndarray`` of strings to an Arkouda ``Strings`` object, whereas the latter converts an Arkouda ``Strings`` object to a NumPy ``ndarray``. As with numeric arrays, if the size of the data exceeds the threshold set by ``arkouda.maxTransferBytes``, the client will raise an exception.

Arkouda currently only supports the HDF5 file format for disk-based I/O. In order to read an array of strings from an HDF5 file, the strings must be stored in an HDF5 ``group`` containing two datasets: ``segments`` (an integer array corresponding to ``offsets`` above) and ``values`` (a ``uint8`` array corresponding to ``bytes`` above). See :ref:`data-preprocessing-label` for more information and guidelines.

Operations
===========

Arkouda ``Strings`` objects support the following operations:

* Indexing with integer, slice, integer ``pdarray``, and boolean ``pdarray`` (see :ref:`indexing-label`)
* Comparison (``==`` and ``!=``) with string literal or other ``Strings`` object of same size
* :ref:`setops-label`, e.g. ``unique`` and ``in1d``
* :ref:`sorting-label`, via ``argsort`` and ``coargsort``
* :ref:`groupby-label`, both alone and in conjunction with numeric arrays
* :ref:`cast-label` to and from numeric arrays
* :ref:`concatenate-label` with other ``Strings``

String-Specific Methods
=======================

Substring search
----------------
  
  .. automethod:: arkouda.Strings.contains
                    
  .. automethod:: arkouda.Strings.startswith
                    
  .. automethod:: arkouda.Strings.endswith

Splitting and joining
---------------------

  .. automethod:: arkouda.Strings.peel
                  
  .. automethod:: arkouda.Strings.rpeel

  .. automethod:: arkouda.Strings.stick

  .. automethod:: arkouda.Strings.lstick

Flattening
----------

Given an array of strings where each string encodes a variable-length sequence delimited by a common substring, flattening offers a method for unpacking the sequences into a flat array of individual elements. A mapping between original strings and new array elements can be preserved, if desired. This method can be used in pipe
  
  .. automethod:: arkouda.Strings.flatten

