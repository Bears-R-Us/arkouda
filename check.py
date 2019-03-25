import numpy as np

import arkouda as ak

ak.v = False
ak.connect(server="localhost", port=5555)

a = np.arange(0,10,1)
b = list(ak.arange(0,10,1))
print(a,b)
c = a == b
print(type(c),c)
print(c.any())

ak.disconnect()

