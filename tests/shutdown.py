#!/usr/bin/env python3                                                         

import sys

import arkouda as ak

print(">>> Shutdown arkouda_server")

ak.verbose = False
if len(sys.argv) > 1:
    ak.connect(server=sys.argv[1], port=sys.argv[2])
else:
    ak.connect()

ak.shutdown()
sys.exit()
