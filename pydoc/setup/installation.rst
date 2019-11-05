.. _installation-label:

#################
Installation
#################

Before installing arkouda, make sure to satisfy all the :ref:`prerequisites-label`, including setting up your environment for Chapel.

*****************
Download
*****************

The easiest way to get arkouda is to download, clone, or fork the `arkouda github repo`_.

.. _arkouda github repo: https://github.com/mhmerrill/arkouda/

*****************
Environment Setup
*****************

1. Ensure that ``CHPL_HOME`` is set and ``$CHPL_HOME/bin`` is in your ``PATH`` (consider adding to your .*rc file).
2. Tell arkouda where to find the HDF5 and ZMQ libraries. Do this by creating or modifying the ``Makefile.paths`` file in the arkouda root directory and adding one or more lines of the form

.. code-block:: bash

  $(eval $(call add-path,/path/to/HDF5/root))
  $(eval $(call add-path,/path/to/ZMQ/root))

However, if you have the Anaconda Python distribution, the HDF5 and ZMQ libraries will be in subdirectories of the Anaconda root directory, so your ``Makefile.paths`` need only contain one line:

.. code-block:: bash

  $(eval $(call add-path,/path/to/Anaconda/root))

Be sure to customize these paths appropriately for your system.

****************
Build the Server
****************

Run ``make`` in the arkouda root directory to build the arkouda server program.

Note: this will take 10-20 minutes, depending on your processor.

We recommend adding the arkouda root directory to your ``PATH`` environment variable.

******************
Install the Client
******************

There are two ways to install the python client. It is available from the Python Package Index (PyPI) with:

.. code-block:: bash

   pip3 install arkouda

If you are planning to contribute to arkouda as a developer, you may wish to install an editable version linked to your local copy of the github repo:

.. code-block:: bash

   pip3 install -e path/to/local/arkouda/repo



****************
Troubleshooting
****************

Legacy classes
=====================

Error: The build fails because ``--legacy-classes`` is not recognized

Cause: A version of Chapel older than 1.20.0 is being used

Solution: Either upgrade Chapel or manually remove the ``--legacy-classes`` flag from the Arkouda ``Makefile``.

Chapel not built for this configuration
==========================================

Error: Build fails with a message stating Chapel was not built for this configuration

Solution: While a full rebuild of Chapel is not required, some additional components must be built with the current environment settings. Do this by setting all your arkouda environment variables as above and running::

  cd $CHPL_HOME
  make

This should build the extra components needed by Chapel to compile arkouda.

Unable to find HDF5 or ZMQ
============================================

Error: Cannot find ``-lzmq`` or ``-lhdf5``

Solution: Ensure the path(s) in the arkouda ``Makefile.paths`` file are valid, and that the files ``lib/libzmq.so`` and ``lib/libhdf5.so`` appear there. If not, try reinstalling HDF5 and/or ZMQ at those locations, or install the Anaconda_ distribution and place the Anaconda root directory in ``Makefile.paths``.

.. _Anaconda: https://www.anaconda.com/distribution/
