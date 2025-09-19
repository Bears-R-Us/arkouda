import sys

from context import arkouda as ak


ak.verbose = False
if len(sys.argv) > 1:
    ak.connect(server=sys.argv[1], port=sys.argv[2])
else:
    ak.connect()

ak.shutdown()
sys.exit()
