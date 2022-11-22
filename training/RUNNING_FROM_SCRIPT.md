# Running Arkouda from a Script

<a id="toc"></a>
## Table of Contents
1. [Overview](#overview)
2. [New Server Startup Flags](#flags)
3. [Example Implementation](#example)

<a id="overview"></a>
## Overview
The purpose of this document is to provide an example process for using a script to automatically run the Arkouda server, connect to the server, perform required actions, then shutdown the server without any user interaction required.


<a id="flags"></a>
## New Server Startup Flags
Two new flags were added to the arkouda_server startup command to help streamline this type of Arkouda usage. Both flags are booleans that default to `false`.

- `--serverInfoNoSplash`
  - This flag replaces the usual splash message in the output of the server startup with the server configuration JSON. This JSON can then be parsed to read the server host, server port, and any other configuration info you may need.
  - An example of the JSON configuration that gets written to `stdout`: 
```json
{
  "arkoudaVersion":"v2022.11.17",
  "chplVersion":"1.28.0",
  "ZMQVersion":"4.3.4",
  "HDF5Version":"1.12.1",
  "serverHostname":"MSI",
  "ServerPort":5555,
  "numLocales":1,
  "numPUs":6,
  "maxTaskPar":6,
  "physicalMemory":13272535040,
  "distributionType":"domain(1,int(64),false)",
  "LocaleConfigs":[{"id":0, "name":"MSI", "numPUs":6, "maxTaskPar":6, "physicalMemory":13272535040}],
  "authenticate":false,
  "logLevel":"INFO",
  "regexMaxCaptures":20,
  "byteorder":"little",
  "autoShutdown":true,
  "serverInfoNoSplash":true,
  "ARROW_VERSION":"7.0.0"
}
```

- `--autoShutdown`
  - This flag toggles whether the arkouda_server will shut itself down when the client disconnects. When set to `true`, `ak.disconnect()` will trigger the server shutdown process.

<a id="example"></a>
## Example Implementation

```python
import arkouda as ak

import subprocess
import json

# Update the below path to point to your arkouda_server
cmd = "/Users/<username>/Documents/git/arkouda/arkouda_server -nl 1 --serverInfoNoSplash=true --autoShutdown=true"
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
# get the output as a string
connect_str = p.stdout.readline()

server_data = json.loads(connect_str)
server, port = server_data['serverHostname'], server_data['ServerPort']

ak.connect(server=server, port=int(port))

# Perform whatever Arkouda actions you need
a = ak.arange(15)
b = ak.array(["This", "is", "an", "example"])

ak.disconnect()
```