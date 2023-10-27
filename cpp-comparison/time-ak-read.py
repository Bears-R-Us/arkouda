import time
import arkouda as ak
ak.connect(<update with server host>)

start = time.time()
a = ak.read('test-file*')
stop = time.time()
print(stop-start)
