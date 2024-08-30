.. _quickstart-label:

#######################
Quickstart
#######################

This guide is intended to instruct users on the installation Arkouda. It will also walk through launching and shutting down the server. For usage information, visit our :ref:`usage-label`

**********************
Install Dependencies
**********************

1. Follow the `Chapel Quickstart Guide <https://chapel-lang.org/docs/usingchapel/QUICKSTART.html>`_
2. Follow the `Anaconda Installation Guide <https://docs.anaconda.com/anaconda/install/index.html>`_

**********************
Install Arkouda
**********************

1. Download either zip file from assets section of the `latest Arkouda release <https://github.com/Bears-R-Us/arkouda/releases/latest>`_
2. Unpack the source files:

   .. substitution-code-block:: bash

       tar xzf arkouda-RELEASE_DATE.tar.gz

3. Change to the arkouda directory

   .. substitution-code-block:: bash

       cd arkouda-RELEASE_DATE

4. Create a conda env from `arkouda-env.yml` and activate it

   .. substitution-code-block:: bash

       conda env create -f arkouda-env.yml
       conda activate arkouda

5. Add your conda env to `Makefile.paths`, so `make` knows where to find dependencies

   .. substitution-code-block:: bash

       echo -e "\$(eval \$(call add-path,$CONDA_PREFIX))" >> Makefile.paths

6. Build Arkouda

   .. substitution-code-block:: bash

       make

7. Test Arkouda functionality

   .. substitution-code-block:: bash

       make test

**********************
Launching the Server
**********************

In a terminal, run the arkouda server program with one locale

You should see a startup message like

.. substitution-code-block:: bash

   $ ./arkouda_server -nl 1
   server listening on tcp://<your_machine>.local:5555
   arkouda server version = |release|
   built with chapel version <chapel_version>
   memory limit = 15461882265
   bytes of memory used = 0

or with authentication turned on 

.. substitution-code-block:: bash

   $ ./arkouda_server -nl 1 --authenticate
   server listening on tcp://<your_machine>:5555?token=<token_string>
   arkouda server version = |release|
   built with chapel version <chapel_version>
   memory limit = 15461882265
   bytes of memory used = 0


The first output line is the most important, because it contains the connection url with the hostname and port required for the client to connect to the server.

******************************
Connect the Python 3 Client
******************************

In another terminal window, launch an interactive Python 3 session, such as ``ipython`` or ``jupyter notebook`` (both included with the Anaconda distribution). To connect to the arkouda server, you must import the arkouda module and call connect with the connection url from the server startup messages. In Python, run

.. code-block:: python

   >>> import arkouda as ak
   # default way to connect is
   >>> ak.connect(connect_url='tcp://node01:5555')
   ...
   connected to tcp://node01:5555
   
Substituting the hostname and port appropriately (defaults are 'localhost' and 5555).

******************************
Shutdown/Disconnect
******************************

If desired, you can disconnect from the arkouda server from a connected client with

.. code-block:: python

   >>> ak.disconnect()

or shutdown 

.. code-block:: python

   >>> ak.shutdown()

This command will delete all server-side arrays and cause the ``arkouda_server`` process in the first terminal to exit.

******************************
Using Arkouda
******************************

Want to learn more about using Arkouda? See the :ref:`usage-label` section for the full list of operations supported on arkouda arrays. These operations are quite composable and can be used to implement more complex algorithms as in the :ref:`examples-label` section.