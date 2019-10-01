.. _prerequisites-label:

############################################
Prerequisites
############################################

*******************
Chapel
*******************
(version 1.20.0 or greater)

The arkouda server application is written in Chapel_, a productive and performant parallel programming language. In order to use arkouda, you must first `download Chapel`_ 1.20.0 or later and build it according to the instructions_ for your platform(s). Below are tips for building Chapel to support arkouda.

LLVM
===================

Arkouda requires Chapel to be built with the LLVM parser enabled. This is accomplished by setting the following environment variable prior to building Chapel::
  
  export CHPL_LLVM=llvm

LLVM comes bundled with Chapel, so it does not require a separate download.

Multi-Locale
===================

Chapel and Arkouda are both designed to be portable, running with minimal reconfiguration on a laptop and a supercomputer. In fact, the developers of arkouda typically implement new functionality on a workstation, test performance on a small cluster, and support users on a massively parallel processing architecture. If you plan to use arkouda or Chapel on a mix of single- and multi-node systems, we recommend building Chapel in multi-locale (i.e. multi-node) mode on all platforms, even single machines.

The Chapel documentation has detailed instructions for `multilocale Chapel execution`_, which are important to observe carefully on multi-node systems. Enabling multilocale execution on a single machine, in our experience, simply requires setting two extra environment variables before building and using Chapel::
  
  export CHPL_COMM=gasnet
  export CHPL_LAUNCHER=smp

*******************************
Python 3 (Anaconda recommended)
*******************************
(version 3.6 or greater)

Currently, the arkouda client is written in Python 3. We recommend using the Anaconda_ Python 3 distribution, with Python 3.6 or greater, because it automatically satisfies the remaining prerequisites.

***************************************
HDF5 and ZMQ (included with Anaconda)
***************************************

Arkouda uses HDF5_ for file I/O and ZMQ_ for server-client communication. Both libraries can either be downloaded and built manually or acquired via a Python package manager. For example, both libraries come pre-installed with the Anaconda_ Python distribution and can be found in the ``include``, ``bin``, and ``lib`` subdirectories of the Anaconda root directory.

*******************************
Numpy (included with Anaconda)
*******************************

Arkouda interoperates with the numerical Python package NumPy, using NumPy data types and supporting conversion between NumPy ``ndarray`` and arkouda ``pdarray`` classes.

The best way to get NumPy is via the Anaconda_ distribution or through a Python package manager like ``pip``.

**********************************************
Pandas (recommended; included with Anaconda)
**********************************************

While Pandas is not required by the arkouda client, some of the arkouda tests use Pandas as a standard to check the correctness of arkouda operations. As with NumPy, the best way to get Pandas is via the Anaconda_ distribution or a Python package manager.

.. _Chapel: https://chapel-lang.org/
.. _download Chapel: https://chapel-lang.org/download.html
.. _instructions: https://chapel-lang.org/download.html
.. _multilocale Chapel execution: https://chapel-lang.org/docs/usingchapel/multilocale.html
.. _Anaconda: https://www.anaconda.com/distribution/
.. _HDF5: https://support.hdfgroup.org/HDF5/
.. _ZMQ: https://zeromq.org/
