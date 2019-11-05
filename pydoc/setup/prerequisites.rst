.. _prerequisites-label:

############################################
Prerequisites
############################################

*******************
Chapel
*******************
(version 1.20.0 or greater)

The arkouda server application is written in Chapel_, a productive and performant parallel programming language. In order to use arkouda, you must first `download the current version of Chapel`_ and build it according to the instructions_ for your platform(s). Below are tips for building Chapel to support arkouda.

Multi-Locale
===================

Chapel and Arkouda are both designed to be portable, running with minimal reconfiguration on a laptop and a supercomputer. In fact, the developers of arkouda typically implement new functionality on a workstation, test performance on a small cluster, and support users on a massively parallel processing architecture. The Chapel documentation has detailed instructions for `multilocale Chapel execution`_, which are important to carefully observe on multi-node systems.

For an individual machine (e.g. a laptop or a workstation), you have two options. The default is single-locale mode, which is also the easiest and most performant. You do not need any special settings to enable this mode; simply build Chapel according to the above instructions. However, if you want your single machine to emulate a multi-node system (e.g. you want to test multi-node functionality on your laptop before moving to a larger system), you can enabling multilocale execution on a single machineby simply setting these environment variables::
  
  export CHPL_COMM=gasnet
  export CHPL_LAUNCHER=smp

and (re)running ``make`` within ``$CHPL_HOME``. Both single- and multi-locale Chapel builds can happily coexist side-by-side. If you have built Chapel with both configurations, you can switch between them by setting ``export CHPL_COMM=none`` or ``export CHPL_COMM=gasnet`` before compiling your Chapel program (e.g. the arkouda server).

*******************************
Python 3 (Anaconda recommended)
*******************************
(version 3.6 or greater)

Currently, the arkouda client is written in Python 3. We recommend using the Anaconda_ Python 3 distribution, with Python 3.6 or greater, because it automatically satisfies the remaining prerequisites.

***************************************
HDF5 and ZMQ (included with Anaconda)
***************************************

Arkouda uses HDF5_ for file I/O and ZMQ_ for server-client communication. Both libraries can either be downloaded and built manually or acquired via a Python package manager. For example, both libraries come pre-installed with the Anaconda_ Python distribution and can be found in the ``include``, ``bin``, and ``lib`` subdirectories of the Anaconda root directory.

Alternatively, running ``pip3 install arkouda`` will also install these dependencies from the PyPI_.

*******************************
Numpy (included with Anaconda)
*******************************

Arkouda interoperates with the numerical Python package NumPy, using NumPy data types and supporting conversion between NumPy ``ndarray`` and arkouda ``pdarray`` classes.

The best way to get NumPy is via the Anaconda_ distribution or from the PyPI_ via ``pip3 install arkouda``.

**********************************************
Pandas (recommended; included with Anaconda)
**********************************************

While Pandas is not required by the arkouda client, some of the arkouda tests use Pandas as a standard to check the correctness of arkouda operations. As with NumPy, the best way to get Pandas is via the Anaconda_ distribution or a the PyPI_.

.. _PyPI: https://pypi.org/
.. _Chapel: https://chapel-lang.org/
.. _download the current version of Chapel: https://chapel-lang.org/download.html
.. _instructions: https://chapel-lang.org/docs/usingchapel/index.html
.. _multilocale Chapel execution: https://chapel-lang.org/docs/usingchapel/multilocale.html
.. _Anaconda: https://www.anaconda.com/distribution/
.. _HDF5: https://support.hdfgroup.org/HDF5/
.. _ZMQ: https://zeromq.org/
