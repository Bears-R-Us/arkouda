#############
Usage
#############

**************
I/O
**************

From disk
=========

All disk-based I/O uses the HDF5 file format. Arkouda is designed to read multiple HDF5 files into a single array, as long as those files all contain a dataset with the same name.

.. autofunction:: arkouda.read_hdf

For convenience, multiple datasets can be read in to create a dictionary of pdarrays.

.. autofunction:: arkouda.read_all


HDF5 files can be queried via the server for dataset names and sizes.

.. autofunction:: arkouda.get_datasets

.. autofunction:: arkouda.ls_hdf

Persistence
===========

Arkouda supports saving pdarrays to HDF5 files. Unfortunately, arkouda cannot yet save to a single HDF5 file from multiple locales and must create one output file per locale.

.. autofunction:: arkouda.pdarray.save

.. autofunction:: arkouda.save_all

These functions allow loading previously saved pdarrays.

.. autofunction:: arkouda.load

.. autofunction:: arkouda.load_all

To/from Python
==============

.. autofunction:: arkouda.array

.. autofunction:: arkouda.pdarray.to_ndarray
