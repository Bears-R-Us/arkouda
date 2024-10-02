# GASNet Development

Certain features of Arkouda require it to be built with GASNet in order to test or debug. Building Arkouda with GASNet requires some additional configuration. Here we walk through the configuration and steps to take for building Arkouda with GASNet.

## Environment Configuration

In order to build Arkouda with GASNet, you must first configure some environment variables. The individual commands are listed below:

```bash
export CHPL_COMM=gasnet
export GASNET_SPAWNFN=L
export GASNET_ROUTE_OUTPUT=0
export CHPL_GASNET_CFG_OPTIONS=--disable-ibv
export GASNET_QUIET=Y
export GASNET_MASTERIP=127.0.0.1
export GASNET_WORKERIP=127.0.0.0

export CHPL_TEST_TIMEOUT=500

export CHPL_RT_OVERSUBSCRIBED=yes
```

It is recommended that you place these into an executable file named `gasnetSetup`. This will allow you to quickly set the required environment variables by executing `source gasnetSetup` from the directory where they file is saved.

## Build Chapel with GASNet

Once your environment is configured, you are ready to build Chapel using GASNet. Run the following:

```bash
cd $CHPL_HOME
make -j 8  # you can bump this up 16 if you have enough memory
``` 

Once complete, Chapel has been built with GASNet.

## Build Arkouda

From your `arkouda` directory run:

```bash
make
```

## Run Arkouda

To run arkouda with multiple locales: 

```bash
./arkouda_server -nl 2
```

If you would like to run with more locales, replace `2` with the number of desired locales. *Please Note that is is not recommended to run on a standard machine with more than 2 locales.*