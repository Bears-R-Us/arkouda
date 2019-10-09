**************
Data I/O
**************

Between client and server
=========================

Arkouda is designed to integrate with NumPy and Pandas, with arkouda handling large, distributed data in parallel while receiving and sending smaller input and output data to/from Python as NumPy ``ndarray`` objects. A common arkouda workflow looks like

1. Load in a large dataset with arkouda
2. Enter or create a small NumPy array with user data to compare against the large dataset
3. Convert the NumPy array to an arkouda array (transferring the data to the server)
4. Run computations that filter or summarize the large dataset
5. Pass the smaller result set back to Python as a NumPy array for plotting or inspection

Below are the two functions that enable both sides of this transfer.

.. autofunction:: arkouda.array

.. autofunction:: arkouda.pdarray.to_ndarray


Large Datasets
=================

Data Preprocessing
------------------

Arkouda is designed to work primarily with columnar data spread across multiple files of non-uniform size. All disk-based I/O uses the HDF5 file format and associates each column of data with an HDF5 dataset present at the root level of all files.

Files are processed in parallel with one file per locale. While HDF5 has an MPI layer for concurrent reading and writing of a single file from multiple nodes, arkouda does not yet support this functionality.

Because most data does not come in HDF5 format, the arkouda developers use arkouda in conjunction with several data preprocessing pipelines. While each dataset requires a unique conversion strategy, all preprocessing should:

* Transpose row-based formats (e.g. CSV) to columns and output each column as an HDF5 dataset
* NOT aggregate input files too aggressively, but keep them separate to enable parallel I/O (hundreds or thousands of files is appropriate, in our experience)
* Convert text to numeric types where possible

Much of this preprocessing can be accomplished with the Pandas ``read*`` functions for ingest and the ``h5py`` module for output. See `this example`_ for ideas.

.. _this example: https://github.com/reuster986/hdflow

Reading HDF5 data from disk
---------------------------

.. autofunction:: arkouda.read_hdf

For convenience, multiple datasets can be read in to create a dictionary of pdarrays.

.. autofunction:: arkouda.read_all


HDF5 files can be queried via the server for dataset names and sizes.

.. autofunction:: arkouda.get_datasets

.. autofunction:: arkouda.ls_hdf

Persisting ``pdarray`` data to disk
-----------------------------------

Arkouda supports saving pdarrays to HDF5 files. Unfortunately, arkouda does not yet support writing to a single HDF5 file from multiple locales and must create one output file per locale.

.. autofunction:: arkouda.pdarray.save

.. autofunction:: arkouda.save_all

Loading persisted arrays from disk
-----------------------------------
These functions allow loading ``pdarray`` data persisted with ``save()`` and ``save_all()``.

.. autofunction:: arkouda.load

.. autofunction:: arkouda.load_all

