**********
Startup
**********

Launch arkouda server
=====================

Follow the :ref:`installation-label` instructions to build the arkouda server program. In a terminal, launch it with

.. code-block:: none

   arkouda_server -nl <numLocales>

Choose a number of locales that is right for your system and data. The ``-h`` flag gives a detailed usage with additional command-line options added by Chapel.

The last line of output from the ``arkouda_server`` command should look like

.. code-block:: none
		
   server listening on node01:5555

Use this hostname and port in the next step to connect to the server.

Connect a Python 3 client
=========================

In Python 3, connect to the arkouda server using the hostname and port shown by the server program (example values shown here)

.. code-block:: python

   >>> import arkouda as ak
   >>> ak.connect('node01', 5555)
   ...
   connected to node01:5555

If the output does not say "connected", then something went wrong (even if the command executes). Check that the hostname and port match what the server printed, and that the hostname is reachable from the machine on which the client is running (e.g. not "localhost" for a remote server)

.. autofunction:: arkouda.connect
