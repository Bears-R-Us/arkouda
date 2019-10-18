/* arkouda server
backend chapel program to mimic ndarray from numpy
This is the main driver for the arkouda server */

use ServerConfig;

use Time only;
use ZMQ only;

use MultiTypeSymbolTable;
use MultiTypeSymEntry;
use MsgProcessing;
use GenSymIO;

proc main() {
    writeln("arkouda server version = ",arkoudaVersion); try! stdout.flush();
    writeln("zeromq version = ", ZMQ.version); try! stdout.flush();
    writeln("makeDistDom.type = ", (makeDistDom(10).type):string); try! stdout.flush();

    var st = new owned SymTab();
    var shutdownServer = false;

    // create and connect ZMQ socket
    var context: ZMQ.Context;
    var socket = context.socket(ZMQ.REP);
    socket.bind("tcp://*:%t".format(ServerPort));
    writeln("server listening on %s:%t".format(serverHostname, ServerPort)); try! stdout.flush();

    var reqCount: int = 0;
    var repCount: int = 0;
    while !shutdownServer {
        // receive requests
        var reqMsg = socket.recv(string);

        // start timer for command processing
        var t1 = Time.getCurrentTime();

        reqCount += 1;

        // shutdown server
        if reqMsg == "shutdown" {
            if v {writeln("reqMsg: ", reqMsg); try! stdout.flush();}
            shutdownServer = true;
            repCount += 1;
            socket.send("shutdown server (%i req)".format(repCount));
            //socket.close(1000); /// error for some reason on close
            break;
        }

        var repMsg: string;
        
        // peel off the command
        var fields = reqMsg.split(1);
        var cmd = fields[1];
        
        if v {
            if cmd == "array" { // has binary data in it's payload
                writeln("reqMsg: ", cmd, " <binary-data>");
            }
            else {
                writeln("reqMsg: ", reqMsg);
            }
            writeln(">>> ",cmd," started at ",t1,"sec");
            try! stdout.flush();
        }

        // parse requests, execute requests, format responses
        select cmd
        {
	    when "lshdf"             {repMsg = lshdfMsg(reqMsg, st);}
	    when "readhdf"           {repMsg = readhdfMsg(reqMsg, st);}
	    when "tohdf"             {repMsg = tohdfMsg(reqMsg, st);}
	    when "array"             {repMsg = arrayMsg(reqMsg, st);}
            when "create"            {repMsg = createMsg(reqMsg, st);}
            when "delete"            {repMsg = deleteMsg(reqMsg, st);}
            when "binopvv"           {repMsg = binopvvMsg(reqMsg, st);}
            when "binopvs"           {repMsg = binopvsMsg(reqMsg, st);}
            when "binopsv"           {repMsg = binopsvMsg(reqMsg, st);}
            when "opeqvv"            {repMsg = opeqvvMsg(reqMsg, st);}
            when "opeqvs"            {repMsg = opeqvsMsg(reqMsg, st);}
            when "efunc"             {repMsg = efuncMsg(reqMsg, st);}
	    when "efunc3vv"          {repMsg = efunc3vvMsg(reqMsg, st);}
	    when "efunc3vs"          {repMsg = efunc3vsMsg(reqMsg, st);}
	    when "efunc3sv"          {repMsg = efunc3svMsg(reqMsg, st);}
	    when "efunc3ss"          {repMsg = efunc3ssMsg(reqMsg, st);}
            when "reduction"         {repMsg = reductionMsg(reqMsg, st);}
  	    when "countReduction"    {repMsg = countReductionMsg(reqMsg, st);}
   	    when "countLocalRdx"     {repMsg = countLocalRdxMsg(reqMsg, st);}
	    when "findSegments"      {repMsg = findSegmentsMsg(reqMsg, st);}
	    when "findLocalSegments" {repMsg = findLocalSegmentsMsg(reqMsg, st);}
	    when "segmentedReduction"{repMsg = segmentedReductionMsg(reqMsg, st);}
	    when "segmentedLocalRdx" {repMsg = segmentedLocalRdxMsg(reqMsg, st);}
            when "arange"            {repMsg = arangeMsg(reqMsg, st);}
            when "linspace"          {repMsg = linspaceMsg(reqMsg, st);}
            when "randint"           {repMsg = randintMsg(reqMsg, st);}
            when "histogram"         {repMsg = histogramMsg(reqMsg, st);}
            when "in1d"              {repMsg = in1dMsg(reqMsg, st);}
            when "unique"            {repMsg = uniqueMsg(reqMsg, st);}
            when "value_counts"      {repMsg = value_countsMsg(reqMsg, st);}
            when "set"               {repMsg = setMsg(reqMsg, st);}
            when "info"              {repMsg = infoMsg(reqMsg, st);}
            when "str"               {repMsg = strMsg(reqMsg, st);}
            when "repr"              {repMsg = reprMsg(reqMsg, st);}
	    when "tondarray"         {repMsg = tondarrayMsg(reqMsg, st);}
            when "[int]"             {repMsg = intIndexMsg(reqMsg, st);}
            when "[slice]"           {repMsg = sliceIndexMsg(reqMsg, st);}
            when "[pdarray]"         {repMsg = pdarrayIndexMsg(reqMsg, st);}
            when "[int]=val"         {repMsg = setIntIndexToValueMsg(reqMsg, st);}
            when "[pdarray]=val"     {repMsg = setPdarrayIndexToValueMsg(reqMsg, st);}            
            when "[pdarray]=pdarray" {repMsg = setPdarrayIndexToPdarrayMsg(reqMsg, st);}            
            when "[slice]=val"       {repMsg = setSliceIndexToValueMsg(reqMsg, st);}            
            when "[slice]=pdarray"   {repMsg = setSliceIndexToPdarrayMsg(reqMsg, st);}
            when "argsort"           {repMsg = argsortMsg(reqMsg, st);}
	    when "coargsort"         {repMsg = coargsortMsg(reqMsg, st);}
	    when "concatenate"       {repMsg = concatenateMsg(reqMsg, st);}
	    when "localArgsort"      {repMsg = localArgsortMsg(reqMsg, st);}
            when "getconfig"         {repMsg = getconfigMsg(reqMsg, st);}
            when "getmemused"        {repMsg = getmemusedMsg(reqMsg, st);}
            when "connect" {
                repMsg = "connected to arkouda server tcp://*:%t".format(ServerPort);
            }
            when "disconnect" {
                repMsg = "disconnected from arkouda server tcp://*:%t".format(ServerPort);
            }
            otherwise {
                if v {writeln("Error: unrecognized command: %s".format(reqMsg)); try! stdout.flush();}
            }
        }
        
        // send responses
        // send count for now
        repCount += 1;
        if v {
	  if cmd == "tondarray" {
              writeln("repMsg:"," <binary-data>");
	  } else {
	    writeln("repMsg:",repMsg);
	  }
	  try! stdout.flush();
	}
        socket.send(repMsg);

        // end timer for command processing
        if v{writeln("<<< ", cmd," took ", Time.getCurrentTime() - t1,"sec"); try! stdout.flush();}
    }

    writeln("requests = ",reqCount," responseCount = ",repCount);
}

