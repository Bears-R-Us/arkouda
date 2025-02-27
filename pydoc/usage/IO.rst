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

.. autofunction:: arkouda.numpy.Strings.to_ndarray

Large Datasets
=================

.. _data-preprocessing-label:

Supported File Formats
----------------------

* HDF5
    * Default File Format
* Parquet
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

Import/Export
=============
Import allows users to import data written by pandas into arkouda. Export allows users to write arkouda data into a format pandas can read. The file formats supported are:

- HDF5
- Parquet

These save formats are customizable and allow for schemas to be created to fit specific needs. As a result, a file written by Arkouda is not always able to be read by other applications. The import/export features of Arkouda allow for files to be reformatted for reading by Pandas and vice versa.

**Import**
Importing data takes a file that was saved using Pandas and reads it into Arkouda. The user is able to specify if they would like to save the result to a file that can be read by Arkouda and/or return the resulting Arkouda object.

**Export**
Export takes a file taht was saved using Arkouda and reads it into Pandas. The user is able to specify if they would like to save the result to a file that can be read by Pandas and/or return the resulting Pandas object.

Note: If the file being read in is Parquet, the resulting file that can be read by Arkouda will also be Parquet. This is also true for HDF5.

This functionality is currently performed on the client and is assuming that dataset sizes are able to be handled in the client due to being written by Pandas. Arkouda natively verifies the size of data before writing it to the client, so exports are limited.

.. autofunction:: arkouda.import_data

.. autofunction:: arkouda.export