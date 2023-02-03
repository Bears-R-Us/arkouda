.. _io-label:

File I/O
======================

Arkouda supports reading and writing files to multiple file formats. 

Arkouda also supports importing files written by Pandas. 

.. toctree::
    :caption: Supported File Formats:
    :maxdepth: 1

    HDF5
    PARQUET
    CSV

Import/Export Support
----------------------
Arkouda supports importing/exporting data in Pandas format. For information, please view the `Import/Export <IMPORT_EXPORT.html>`_ documentation.

.. toctree::
    :hidden:
    :maxdepth: 1

    IMPORT_EXPORT

General I/O API
----------------

Arkouda supplies functions for general I/O interactions. These functions allow for writing 1 or more Arkouda objects and reading data into Arkouda objects.

Write
^^^^^^
- :py:func:`arkouda.io.to_parquet`
- :py:func:`arkouda.io.to_hdf`
- :py:func:`arkouda.io.to_csv`
- :py:func:`arkouda.io.save_all`

Read
^^^^^
- :py:func:`arkouda.io.load`
- :py:func:`arkouda.io.load_all`
- :py:func:`arkouda.io.read_parquet`
- :py:func:`arkouda.io.read_hdf`
- :py:func:`arkouda.io.read_csv`
- :py:func:`arkouda.io.read`

`ls` Functionality
^^^^^^^^^^^^^^^^^^^
These functions allow the user to access a list of datasets/columns stored in the provided file.

- :py:func:`arkouda.io.ls`
- :py:func:`arkouda.io.ls_csv`
- :py:func:`arkouda.io.get_datasets`
- :py:func:`arkouda.io.get_columns`
