module TransferMsg
{
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use Message;
    use Reflection;
    use CTypes;
    use ZMQ;
    use List;
    use GenSymIO;

    use SegmentedString;
    use SegmentedArray;

    enum ObjType {
      ARRAYVIEW=0,
      PDARRAY=1,
      STRINGS=2,
      SEGARRAY=3,
      CATEGORICAL=4
    };
    
    proc sendArrMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var objType: ObjType = msgArgs.getValueOf("objType").toUpper(): ObjType;

      select objType {
        when ObjType.ARRAYVIEW {
          // call handler for arrayview write msg
          //arrayView_tohdfMsg(msgArgs, st);
        }
        when ObjType.PDARRAY {
          // call handler for pdarray write
          pdarrayTransfer(msgArgs, st);
        }
        when ObjType.STRINGS {
          // call handler for strings write
          stringsTransfer(msgArgs, st);
        }
        when ObjType.SEGARRAY {
          // call handler for segarray write
          //segarrayTransfer(msgArgs, st);
        }
        when ObjType.CATEGORICAL {
          //categoricalTransfer(msgArgs, st);
        }
        otherwise {
          var errorMsg = "Unable to transfer object type %s.".format(objType);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
      
      return new MsgTuple("Array sent", MsgType.NORMAL);
    }

    proc pdarrayTransfer(msgArgs: borrowed MessageArgs, st: borrowed SymTab) throws {
      var hostname = msgArgs.getValueOf("hostname");
      var port = msgArgs.getValueOf("port");
      
      var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("arg1"), st);
      select gEnt.dtype {
        when DType.Int64 {
          var e = toSymEntry(gEnt, int);
          sendData(e, hostname, port);
        } when DType.UInt64 {
          var e = toSymEntry(gEnt, uint);
          sendData(e, hostname, port);
        } when DType.Float64 {
          var e = toSymEntry(gEnt, real);
          sendData(e, hostname, port);
        } when DType.Bool {
          var e = toSymEntry(gEnt, bool);
          sendData(e, hostname, port);
        }
      }
    }

    proc stringsTransfer(msgArgs: borrowed MessageArgs, st: borrowed SymTab) throws {
      var hostname = msgArgs.getValueOf("hostname");
      var port = msgArgs.getValueOf("port");
      
      var entry:SegStringSymEntry = toSegStringSymEntry(st.lookup(msgArgs.getValueOf("values")));
      var segString = new SegString("", entry);
      sendData(segString.values, hostname, port);
      sendData(segString.offsets, hostname, port);
    }

    proc receiveArrMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var hostname = msgArgs.getValueOf("hostname");
      var port = msgArgs.getValueOf("port");

      // tuple (dsetName, item type, id)
      var rnames: list((string, string, string)); 

      // send number of locales so that the sender knows how to chunk data
      sendLocaleCount(port);

      var (size, typeString, nodeNames) = receiveSetupInfo(hostname, port);

      var rname = st.nextName();
      if typeString == "int(64)" {
        var entry = new shared SymEntry(size, int);
        receiveData(entry.a, nodeNames, port);
        st.addEntry(rname, entry);
        rnames.append(("single", "pdarray", rname));
      } else if typeString == "uint(64)" {
        var entry = new shared SymEntry(size, uint);
        receiveData(entry.a, nodeNames, port);
        st.addEntry(rname, entry);
        rnames.append(("single", "pdarray", rname));
      } else if typeString == "real(64)" {
        var entry = new shared SymEntry(size, real);
        receiveData(entry.a, nodeNames, port);
        st.addEntry(rname, entry);
        rnames.append(("single", "pdarray", rname));
      } else if typeString == "bool" {
        var entry = new shared SymEntry(size, bool);
        receiveData(entry.a, nodeNames, port);
        st.addEntry(rname, entry);
        rnames.append(("single", "pdarray", rname));
      } else if typeString == "uint(8)" {
        // this is a strings, so we know there are going to be two receives
        var values = new shared SymEntry(size, uint(8));
        receiveData(values.a, nodeNames, port);
        sendLocaleCount(port);
        var (offSize, _, _) = receiveSetupInfo(hostname, port);
        var offsets = new shared SymEntry(offSize, int);
        receiveData(offsets.a, nodeNames, port);
        var stringsEntry = assembleSegStringFromParts(offsets, values, st);
        //getSegString(offsets.a, values.a, st);
        rnames.append(("", "seg_string", "%s+%t".format(stringsEntry.name, stringsEntry.nBytes)));
      }

      var transferErrors: list(string);
      var repMsg = _buildReadAllMsgJson(rnames, false, 0, transferErrors, st);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc sendData(e, hostname:string, port) throws {
      // get locale count so that we can chunk data
      var localeCount = receiveLocaleCount(hostname, port:string);

      // these are the subdoms of the receiving node
      const otherSubdoms = getSubdoms(e.a, localeCount);
      const hereSubdoms = getSubdoms(e.a, Locales.size);

      // size of this is how many messages are going to be sent
      const intersections = getIntersections(otherSubdoms, hereSubdoms);

      const names = Locales.name;
      const namesThatWillSendMessages = getNodeList(intersections, hereSubdoms, names);
      const ports = getPorts(namesThatWillSendMessages, port:int);

      sendSetupInfo(port, e.a, names);

      coforall loc in Locales do on loc {
        var locdom = e.a.localSubdomain();
        // TODO: Parallelize by using different ports
        for (i,p) in zip(intersections, ports) {
          const intersection = getIntersection(locdom, i);
          if intersection.size > 0 {
            sendArrChunk(p, e.a, intersection);
          }
        }
      }
    }

    proc receiveData(A: [] ?t, nodeNames, port) throws {
      const hereSubdoms = getSubdoms(A, Locales.size);
      const otherSubdoms = getSubdoms(A, nodeNames.size);
      // nodeNames lines up with intersections
      const intersections = getIntersections(hereSubdoms, otherSubdoms);

      const nodesToReceiveFrom = getNodeList(intersections, otherSubdoms, nodeNames);
      const ports = getPorts(nodesToReceiveFrom, port:int);

      coforall loc in Locales do on loc {
        //TODO: parallelize
        for (intersection,name,p) in zip(intersections, nodesToReceiveFrom, ports) {
          const intrsct = getIntersection(A.localSubdomain(), intersection);
          if intrsct.size > 0 {
            receiveArrChunk(p, name, A, intersection);
          }
        }
      }
    }

    proc sendLocaleCount(port) throws {
      var context: Context;
      var socket = context.socket(ZMQ.PUSH);
      socket.bind("tcp://*:"+port);

      // first send the size
      socket.send(Locales.size:bytes);
    }

    proc receiveLocaleCount(hostname:string, port:string) throws {
      var context: Context;
      var socket = context.socket(ZMQ.PULL);
      if hostname == here.name then
        socket.connect("tcp://localhost:"+port:string);
      else
        socket.connect("tcp://"+hostname+":"+port:string);
      return socket.recv(bytes):int;
    }

    proc sendSetupInfo(port:string, A: [] ?t, names) throws {
      var context: Context;
      var socket = context.socket(ZMQ.PUSH);
      socket.bind("tcp://*:"+port);

      // first send the size
      socket.send(A.size:bytes);

      // next send the type
      socket.send(t:string);

      // next, send the names of the nodes as a string
      var namesString = "";
      for i in names.domain {
        if i != names.domain.high then
          namesString +=names[i] + " ";
        else
          namesString += names[i];
      }
      socket.send(namesString);
    }

    proc receiveSetupInfo(hostname, port) throws {
      var context: Context;
      var socket = context.socket(ZMQ.PULL);
      if hostname == here.name then
        socket.connect("tcp://localhost:"+port:string);
      else
        socket.connect("tcp://"+hostname+":"+port:string);
      var size = socket.recv(bytes):int;
      var typeString = socket.recv(string);

      // get names of nodes to receive from in order
      var nodeNamesStr = socket.recv(string);
      var nodeNames = nodeNamesStr.split(" ");
      return (size, typeString, nodeNames);
    }

    proc sendArrChunk(port: int, A: [] ?t, intersection: domain(1)) throws {
      var context: Context;
      var socket = context.socket(ZMQ.PUSH);
      socket.bind("tcp://*:"+port:string);
      const size = intersection.size*c_sizeof(t):int;
      var locBuff = createBytesWithBorrowedBuffer(c_ptrTo(A[intersection.low]):c_ptr(uint(8)), size, size);
      socket.send(locBuff);
    }

    proc receiveArrChunk(port, hostname:string, A: [] ?t, intersection: domain(1)) throws {
      var context: Context;
      var socket = context.socket(ZMQ.PULL);
      // if the sending and receiving nodes are the same (sending to self), use localhost
      if hostname == here.name then
        socket.connect("tcp://localhost:"+port:string);
      else
        socket.connect("tcp://"+hostname+":"+port:string);
      var locData = socket.recv(bytes);
      var locArr = makeArrayFromPtr(locData.c_str():c_void_ptr:c_ptr(t), intersection.size:uint);
      A[intersection] = locArr;
    }

    proc getPorts(nodeNames, port) throws {
      // TODO: this could use less ports if we only incremented
      //       duplicate names, but starting with the easier way
      //       of just unique ports for everything
      var ports = [i in 0..#nodeNames.size] port + i;
      return ports;
    }

    proc getNodeList(intersects, subdoms, nodeNames) throws {
      var nodeList: [0..#intersects.size] string;
      for (subdom, nodeName) in zip(subdoms, nodeNames) {
        for (i, val) in zip(intersects, nodeList) {
          const intersection = getIntersection(i, subdom);
          if intersection.size > 0 {
            val = nodeName;
          }
        }
      }
      return nodeList;
    }

    proc getIntersection(d1: domain(1), d2: domain(1)) {
      var low = max(d1.low, d2.low);
      var high = min(d1.high, d2.high);
      return {low..high};
    }

    proc getIntersections(sendingDoms, receivingDoms) {
      var intersections: list(domain(1));
      for d1 in sendingDoms {
        for d2 in receivingDoms {
          const intersection = getIntersection(d1, d2);
          if intersection.size > 0 then
            intersections.append(intersection);
        }
      }
      return intersections;
    }

    proc getSubdoms(A: [] ?t, size:int) {
      var subdoms: [0..#size] domain(1);
      for i in 0..#size {
        subdoms[i] = getBlock(A.domain.low, A.domain.high, size, i);
      }
      return subdoms;
    }
    proc getBlock(const lo: int, const hi:int, const numlocs: int, const locid: int) {
      type idxType = int;
      var inds: range(idxType);
      const numelems = hi - lo + 1;
      const (blo, bhi) = _computeBlock(numelems, numlocs, chpl__tuplify(locid)(0),
                                       hi, lo, lo);
      inds = blo..bhi;
      return inds;
    }

    proc _computeBlock(numelems, numblocks, blocknum, wayhi,
                       waylo=0:wayhi.type, lo=0:wayhi.type) {
      if numelems == 0 then
        return (1:lo.type, 0:lo.type);

      const blo =
        if blocknum == 0 then waylo
        else lo + intCeilXDivByY(numelems:uint * blocknum:uint, numblocks:uint):lo.type;
      const bhi =
        if blocknum == numblocks - 1 then wayhi
        else lo + intCeilXDivByY(numelems:uint * (blocknum+1):uint, numblocks:uint):lo.type - 1;

      return (blo, bhi);
    }

    proc intCeilXDivByY(x, y) {
      return 1 + (x - 1)/y;
    }

    proc bytesToLocArray(size:int, type t, ref data:bytes) throws {
      var res = makeArrayFromPtr(data.c_str():c_void_ptr:c_ptr(t), size:uint);
      return res;
    }
    
    use CommandMap;
    registerFunction("sendArray", sendArrMsg, getModuleName());
    registerFunction("receiveArray", receiveArrMsg, getModuleName());
}
