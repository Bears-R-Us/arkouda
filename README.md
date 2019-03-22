# arkouda
arkouda python/chapel package

 * requires chapel 1.18.0
 * requires zeromq version >= 4.2.5, tested with 4.2.5 and 4.3.1
 * setup your CHPL_HOME env variable and source $CHPL_HOME/util/setchplenv.bash
 * compile arkouda_server.chpl
 * don't forget the --fast flag on the compile line
 * you may also need to use -I to find zmq.h and -L to find libzmq.a
```bash
chpl --fast arkouda_server.chpl
```
 * startup the arkouda_server
 * defaults to port 5555
```bash
./arkouda_server
```
 * config var on the commandline
  * --v=true/false to turn on/off verbose messages from server
  * --ServerPort=5555
 * or you could run it this way if you don't want as many messages
and a different port to be used
```bash
./arkouda_server --ServerPort=5555 --v=false
```
 * in the same directory in a different terminal window
 * run the ak_test.py python3 program
 * this program just does a couple things and calls shutdown for the server
 * edit the server and port in the script to something other than the
default if you ran the server on a different server or port
```bash
./ak_test.py
```
or
```bash
python3 ak_test.py
```
 * This also works fine from a jupyter notebook
 * there is an included Jupyter notebook called test_arkouda.ipynb

