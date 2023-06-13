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
    use Map;
    use ServerErrors;
    use ServerConfig;
    use ServerErrorStrings;

    use SegmentedString;

    proc sendDataFrameSetupInfo(port:string, numColumns: int, elements: string) throws {
      var context: Context;
      var socket = context.socket(ZMQ.PUSH);
      socket.bind("tcp://*:"+port);

      // first, send the number of columns
      socket.send(numColumns);

      // next, send the obj type string
      socket.send(elements);
    }

    proc receiveDataFrameSetupInfo(hostname: string, port:string) throws {
      var context: Context;
      var socket = context.socket(ZMQ.PULL);
      if hostname == here.name then
        socket.connect("tcp://localhost:"+port:string);
      else
        socket.connect("tcp://"+hostname+":"+port:string);
      var numColumns = socket.recv(int);
      
      var objString = socket.recv(string);
      var objNames = objString.split(" ");

      return (numColumns, objNames);
    }
    
    proc sendDataFrameMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      param pn = Reflection.getRoutineName();
      
      var hostname = msgArgs.getValueOf("hostname");
      var port = msgArgs.getValueOf("port");
      
      var numColumns = msgArgs.get("size").getIntValue();
      var eleList = msgArgs.get("columns").getList(numColumns);

      var localeCount = receiveLocaleCount(hostname, port:string);
      
      sendDataFrameSetupInfo(port:string, numColumns, eleList: string);

      for ele in eleList { 
        var ele_parts = ele.split("+");
        ref col_name = ele_parts[1];
        if ele_parts[0] == "Categorical" {
          ref codes_name = ele_parts[2];
          ref categories_name = ele_parts[3];

          var gCode: borrowed GenSymEntry = getGenericTypedArrayEntry(codes_name, st);
          var code_vals = toSymEntry(gCode, int);
        }
        else if ele_parts[0] == "Strings"{
        }
        else if ele_parts[0] == "pdarray" || ele_parts[0] == "IPv4" || 
          ele_parts[0] == "Fields" || ele_parts[0] == "Datetime" || ele_parts[0] == "BitVector"{
          var gCol: borrowed GenSymEntry = getGenericTypedArrayEntry(ele_parts[2], st);
          select (gCol.dtype) {
            when (DType.Int64) {
              var col_vals = toSymEntry(gCol, int);

              var (intersections, ports, names) = calculateSetupInfo(col_vals, localeCount, port);
              sendSetupInfo(port:string, col_vals.a, names, "pdarray", localeCount);
              sendData(col_vals, hostname, intersections, ports);
            }
            when (DType.UInt64) {
              var col_vals = toSymEntry(gCol, uint);
            }
            when (DType.Bool) {
              var col_vals = toSymEntry(gCol, bool);
            }
            when (DType.Float64){
              var col_vals = toSymEntry(gCol, real);
            }
            otherwise {
              var errorMsg = notImplementedError(pn,dtype2str(gCol.dtype));
              throw new IllegalArgumentError(errorMsg);
            }
          }
        }
        else if ele_parts[0] == "SegArray" {
          ref segments_name = ele_parts[2];
          ref values_name = ele_parts[3];

          var gSeg: borrowed GenSymEntry = getGenericTypedArrayEntry(segments_name, st);
          var segments = toSymEntry(gSeg, int);
          var gVal: borrowed GenSymEntry = getGenericTypedArrayEntry(values_name, st);
          select(gVal.dtype){
            when(DType.Int64){
              var values = toSymEntry(gVal, int);
            }
            when(DType.UInt64){
              var values = toSymEntry(gVal, uint);
            }
            when(DType.Float64){
              var values = toSymEntry(gVal, real);
            }
            when(DType.Bool){
              var values = toSymEntry(gVal, bool);
            }
            otherwise {
              var errorMsg = notImplementedError(pn,dtype2str(gVal.dtype));
              throw new IllegalArgumentError(errorMsg);
            }
          }
        }
        else {
          var errorMsg = notImplementedError(pn, ele_parts[0]);
          throw new IllegalArgumentError(errorMsg);
        }
      }
      return new MsgTuple("DataFrame sent", MsgType.NORMAL);
    }
    
    proc receiveDataFrameMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var hostname = msgArgs.getValueOf("hostname");
      var port = msgArgs.getValueOf("port");

      // send number of locales so that the sender knows how to chunk data
      sendLocaleCount(port);

      var (numColumns, objNames) = receiveDataFrameSetupInfo(hostname, port);
      var rnames: list((string, ObjType, string)); 

      for (i, obj) in zip(0..#objNames.size, objNames) {
        var objParts = obj.split("+");
        ref currObjType = objParts[0];
        var rname = st.nextName();
        if currObjType == "pdarray" {
          var (size, typeString, nodeNames, _) = receiveSetupInfo(hostname, port);
          var entry = new shared SymEntry(size, int);
          receiveData(entry.a, nodeNames, port);
          rnames.append((i:string, ObjType.PDARRAY, rname));
          st.addEntry(rname, entry);
        }
      }
      
      var transferErrors: list(string);
      var repMsg = _buildReadAllMsgJson(rnames, false, 0, transferErrors, st);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

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
          segarrayTransfer(msgArgs, st);
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
      // get locale count so that we can chunk data
      var localeCount = receiveLocaleCount(hostname, port:string);
      select gEnt.dtype {
        when DType.Int64 {
          var e = toSymEntry(gEnt, int);

          var (intersections, ports, names) = calculateSetupInfo(e, localeCount, port);
          sendSetupInfo(port:string, e.a, names, "pdarray", localeCount);
          sendData(e, hostname, intersections, ports);
        } when DType.UInt64 {
          var e = toSymEntry(gEnt, uint);

          var (intersections, ports, names) = calculateSetupInfo(e, localeCount, port);
          sendSetupInfo(port:string, e.a, names, "pdarray", localeCount);
          sendData(e, hostname, intersections, ports);
        } when DType.Float64 {
          var e = toSymEntry(gEnt, real);

          var (intersections, ports, names) = calculateSetupInfo(e, localeCount, port);
          sendSetupInfo(port:string, e.a, names, "pdarray", localeCount);
          sendData(e, hostname, intersections, ports);
        } when DType.Bool {
          var e = toSymEntry(gEnt, bool);

          var (intersections, ports, names) = calculateSetupInfo(e, localeCount, port);
          sendSetupInfo(port:string, e.a, names, "pdarray", localeCount);
          sendData(e, hostname, intersections, ports);
        }
      }
    }

    proc stringsTransfer(msgArgs: borrowed MessageArgs, st: borrowed SymTab) throws {
      var hostname = msgArgs.getValueOf("hostname");
      var port = msgArgs.getValueOf("port");
      
      var entry:SegStringSymEntry = toSegStringSymEntry(st.lookup(msgArgs.getValueOf("values")));
      var segString = new SegString("", entry);
      // get locale count so that we can chunk data
      var localeCount = receiveLocaleCount(hostname, port:string);

      {
        var (intersections, ports, names) = calculateSetupInfo(segString.values, localeCount, port);
        sendSetupInfo(port:string, segString.values.a, names, "string", localeCount);
        sendData(segString.values, hostname, intersections, ports);
      }

      {
        var (intersections, ports, names) = calculateSetupInfo(segString.offsets, localeCount, port);
        sendSetupInfo(port:string, segString.offsets.a, names, "pdarray", localeCount);
        sendData(segString.offsets, hostname, intersections, ports);
      }
    }

    proc segarrayTransfer(msgArgs: borrowed MessageArgs, st: borrowed SymTab) throws {
      var hostname = msgArgs.getValueOf("hostname");
      var port = msgArgs.getValueOf("port");

      var segarr = msgArgs.getValueOf("seg_name");
      var dType = str2dtype(msgArgs.getValueOf("dtype"));
      // get locale count so that we can chunk data
      var localeCount = receiveLocaleCount(hostname, port:string);

      var segments = toSymEntry(toGenSymEntry(st.lookup(msgArgs.getValueOf("segments"))), int);

      select dType {
        when (DType.Int64) {
          var values = toSymEntry(toGenSymEntry(st.lookup(msgArgs.getValueOf("values"))), int);

          {
            var (intersections, ports, names) = calculateSetupInfo(values, localeCount, port);
            sendSetupInfo(port:string, values.a, names, "seg_array", localeCount);
            sendData(values, hostname, intersections, ports);
          }

          {
            var (intersections, ports, names) = calculateSetupInfo(segments, localeCount, port);
            sendSetupInfo(port:string, segments.a, names, "seg_array", localeCount);
            sendData(segments, hostname, intersections, ports);
          }
        } when (DType.UInt64) {
          var values = toSymEntry(toGenSymEntry(st.lookup(msgArgs.getValueOf("values"))), uint);
        } when (DType.Float64) {
          var values = toSymEntry(toGenSymEntry(st.lookup(msgArgs.getValueOf("values"))), real);
        } when (DType.Bool) {
          var values = toSymEntry(toGenSymEntry(st.lookup(msgArgs.getValueOf("values"))), bool);
        }
        otherwise {
          throw getErrorWithContext(
                                    msg="Unsupported SegArray DType %s".format(dtype2str(dType)),
                                    lineNumber=getLineNumber(),
                                    routineName=getRoutineName(), 
                                    moduleName=getModuleName(),
                                    errorClass="IllegalArgumentError");
        }
      }
    }

    proc receiveArrMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
      var hostname = msgArgs.getValueOf("hostname");
      var port = msgArgs.getValueOf("port");

      // tuple (dsetName, item type, id)
      var rnames: list((string, ObjType, string)); 

      // send number of locales so that the sender knows how to chunk data
      sendLocaleCount(port);

      var (size, typeString, nodeNames, objType) = receiveSetupInfo(hostname, port);

      if objType == "string" {
        // this is a strings, so we know there are going to be two receives
        var values = new shared SymEntry(size, uint(8));
        receiveData(values.a, nodeNames, port);
        var (offSize, _, _, _) = receiveSetupInfo(hostname, port);
        var offsets = new shared SymEntry(offSize, int);
        receiveData(offsets.a, nodeNames, port);
        var stringsEntry = assembleSegStringFromParts(offsets, values, st);
        //getSegString(offsets.a, values.a, st);
        rnames.append(("", ObjType.STRINGS, "%s+%t".format(stringsEntry.name, stringsEntry.nBytes)));
      } else if objType == "seg_array" {
        var rtnMap: map(string, string) = new map(string, string);
        if typeString == "int(64)" {
          var entry = new shared SymEntry(size, int);
          var vname = st.nextName();
          receiveData(entry.a, nodeNames, port);

          var (segSize, _, _, _) = receiveSetupInfo(hostname, port);
          var segments = new shared SymEntry(segSize, int);
          var sname = st.nextName();
          receiveData(segments.a, nodeNames, port);

          st.addEntry(sname, segments);
          st.addEntry(vname, entry);
          
          rtnMap.add("segments", "created " + st.attrib(sname));
          rtnMap.add("values", "created " + st.attrib(vname));
        } else if typeString == "uint(64)" {
          var entry = new shared SymEntry(size, uint);
          var vname = st.nextName();
          receiveData(entry.a, nodeNames, port);

          var (segSize, _, _, _) = receiveSetupInfo(hostname, port);
          var segments = new shared SymEntry(segSize, int);
          var sname = st.nextName();
          st.addEntry(sname, segments);
          st.addEntry(vname, entry);
          
          rtnMap.add("segments", "created " + st.attrib(sname));
          rtnMap.add("values", "created " + st.attrib(vname));
        } else if typeString == "real(64)" {
          var entry = new shared SymEntry(size, real);
          var vname = st.nextName();
          receiveData(entry.a, nodeNames, port);

          var (segSize, _, _, _) = receiveSetupInfo(hostname, port);
          var segments = new shared SymEntry(segSize, int);
          var sname = st.nextName();
          receiveData(segments.a, nodeNames, port);
          
          st.addEntry(sname, segments);
          st.addEntry(vname, entry);
          
          rtnMap.add("segments", "created " + st.attrib(sname));
          rtnMap.add("values", "created " + st.attrib(vname));
        } else if typeString == "bool" {
          var entry = new shared SymEntry(size, bool);
          receiveData(entry.a, nodeNames, port);
          var vname = st.nextName();

          var (segSize, _, _, _) = receiveSetupInfo(hostname, port);
          var segments = new shared SymEntry(segSize, int);
          receiveData(segments.a, nodeNames, port);
          var sname = st.nextName();
          
          st.addEntry(sname, segments);
          st.addEntry(vname, entry);
          
          rtnMap.add("segments", "created " + st.attrib(sname));
          rtnMap.add("values", "created " + st.attrib(vname));
        }
        rnames.append(("", ObjType.SEGARRAY, "%jt".format(rtnMap)));
      } else {
        var rname = st.nextName();
        if typeString == "int(64)" {
          var entry = new shared SymEntry(size, int);
          receiveData(entry.a, nodeNames, port);
          st.addEntry(rname, entry);
        } else if typeString == "uint(64)" {
          var entry = new shared SymEntry(size, uint);
          receiveData(entry.a, nodeNames, port);
          st.addEntry(rname, entry);
        } else if typeString == "real(64)" {
          var entry = new shared SymEntry(size, real);
          receiveData(entry.a, nodeNames, port);
          st.addEntry(rname, entry);
        } else if typeString == "bool" {
          var entry = new shared SymEntry(size, bool);
          receiveData(entry.a, nodeNames, port);
          st.addEntry(rname, entry);
        }
        rnames.append(("", ObjType.PDARRAY, rname));
      }

      var transferErrors: list(string);
      var repMsg = _buildReadAllMsgJson(rnames, false, 0, transferErrors, st);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc sendData(e, hostname:string, intersections, ports) throws {
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

    proc calculateSetupInfo(e, localeCount, port) throws {
      // these are the subdoms of the receiving node
      const otherSubdoms = getSubdoms(e.a, localeCount);
      const hereSubdoms = getSubdoms(e.a, Locales.size);

      // size of this is how many messages are going to be sent
      const intersections = getIntersections(otherSubdoms, hereSubdoms);

      const names = Locales.name;
      const namesThatWillSendMessages = getNodeList(intersections, hereSubdoms, names);
      const ports = getPorts(namesThatWillSendMessages, port:int);
      return (intersections, ports, names);
    }
    
    proc sendSetupInfo(port:string, A: [] ?t, names, objType: string, localeCount) throws {
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
      socket.send(objType);
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
      var objType = socket.recv(string);
      return (size, typeString, nodeNames, objType);
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
    registerFunction("sendDataframe", sendDataFrameMsg, getModuleName());
    registerFunction("receiveDataframe", receiveDataFrameMsg, getModuleName());
}
