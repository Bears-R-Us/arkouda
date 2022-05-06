.. _IO-label:

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

Below are the functions that enable both sides of this transfer.

.. autofunction:: arkouda.array
		  
.. autofunction:: arkouda.pdarray.to_ndarray

.. autofunction:: arkouda.Strings.to_ndarray

Large Datasets
=================

.. _data-preprocessing-label:

Supported File Formats
----------------------

* HDF5
    * Default File Format
* Parquet
    * Optional
    * Requires `pyarrow`

Data Preprocessing
------------------

Arkouda is designed to work primarily with columnar data spread across multiple files of non-uniform size. All disk-based I/O uses HDF5 or Parquet file format and associates each column of data with an HDF5/Parquet dataset present at the root level of all files.

Files are processed in parallel with one file per locale. While HDF5 has an MPI layer for concurrent reading and writing of a single file from multiple nodes, arkouda does not yet support this functionality.

Because most data does not come in HDF5/Parquet format, the arkouda developers use arkouda in conjunction with several data preprocessing pipelines. While each dataset requires a unique conversion strategy, all preprocessing should:

* Transpose row-based formats (e.g. CSV) to columns and output each column as an HDF5 dataset
* NOT aggregate input files too aggressively, but keep them separate to enable parallel I/O (hundreds or thousands of files is appropriate, in our experience)
* Convert text to numeric types where possible

Much of this preprocessing can be accomplished with the Pandas ``read*`` functions for ingest and the ``h5py`` or ``pyarrow`` module for output. See `this example`_ for ideas.

.. _this example: https://github.com/reuster986/hdflow

Reading data from disk
---------------------------

.. autofunction:: arkouda.read

For convenience, multiple datasets can be read in to create a dictionary of pdarrays.

.. autofunction:: arkouda.read_all


HDF5/Parquet files can be queried via the server for dataset names and sizes.

.. autofunction:: arkouda.get_datasets

.. autofunction:: arkouda.ls_any

Persisting ``pdarray`` data to disk
-----------------------------------

Arkouda supports saving pdarrays to HDF5/Parquet files. Unfortunately, arkouda does not yet support writing to a single HDF5 file from multiple locales and must create one output file per locale.

.. autofunction:: arkouda.pdarray.save

.. autofunction:: arkouda.save_all

Loading persisted arrays from disk
-----------------------------------
These functions allow loading ``pdarray`` data persisted with ``save()`` and ``save_all()``.

.. autofunction:: arkouda.load

.. autofunction:: arkouda.load_all

Persisting ``DataFrame`` data to disk
-------------------------------------
Arkouda supports saving ``DataFrame`` objects to HDF5/Parquet files. This is done by creating a dictionary that maps the column name to the pdarray containing the column data. The column names are treated as datasets in the file.

.. autofunction:: arkouda.DataFrame.save_table

Loading persisted DataFrame data from disk
-------------------------------------------
This functionality allows the columns be loaded as datasets, which creates a mapping of column names to column data. This structure is supported by the ``DataFrame`` constructor and is used to reconstruct the ``DataFrame``

.. autofunction:: arkouda.DataFrame.load_table