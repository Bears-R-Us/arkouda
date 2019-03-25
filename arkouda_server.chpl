// arkouda server
// backend chapel program to mimic ndarray from numpy
// This is the main driver for the arkouda server
//
use ServerConfig;

use Time;
use ZMQ;
use MultiTypeSymbolTable;
use MultiTypeSymEntry;
use MsgProcessing;

proc main() {
    writeln("arkouda server version = ",arkouda_version); try! stdout.flush();
    writeln("zeromq server on port %t".format(ServerPort)); try! stdout.flush();
    writeln("zeromq version = ", ZMQ.version); try! stdout.flush();
    writeln("makeDistDom.type = ", (makeDistDom(10).type):string); try! stdout.flush();

    var st = new owned SymTab();
    var shutdown_server = false;

    // create and connect ZMQ socket
    var context: Context;
    var socket = context.socket(ZMQ.REP);
    socket.bind("tcp://*:%t".format(ServerPort));

    var req_count: int = 0;
    var rep_count: int = 0;
    while !shutdown_server {
        // receive requests
        var req_msg = socket.recv(string);

        // start timer for command processing
        var t1 = getCurrentTime();

        req_count += 1;
        if v {writeln("received: ", req_msg); try! stdout.flush();}

        // shutdown server
        if req_msg == "shutdown" {
            shutdown_server = true;
            rep_count += 1;
            socket.send("shutdown server (%i req)".format(rep_count));
            //socket.close(1000); /// error for some reason on close
            break;
        }

        var repMsg: string;
        
        // peel off the command
        var fields = req_msg.split(1);
        var cmd = fields[1];
        if v {writeln(">>> ",cmd); try! stdout.flush();}
        // parse requests, execute requests, format responses
        select cmd
        {
            when "create"            {repMsg = createMsg(req_msg, st);}
            when "delete"            {repMsg = deleteMsg(req_msg, st);}
            when "binopvv"           {repMsg = binopvvMsg(req_msg, st);}
            when "binopvs"           {repMsg = binopvsMsg(req_msg, st);}
            when "binopsv"           {repMsg = binopsvMsg(req_msg, st);}
            when "opeqvv"            {repMsg = opeqvvMsg(req_msg, st);}
            when "opeqvs"            {repMsg = opeqvsMsg(req_msg, st);}
            when "efunc"             {repMsg = efuncMsg(req_msg, st);}
            when "reduction"         {repMsg = reductionMsg(req_msg, st);}
            when "arange"            {repMsg = arangeMsg(req_msg, st);}
            when "linspace"          {repMsg = linspaceMsg(req_msg, st);}
            when "randint"           {repMsg = randintMsg(req_msg, st);}
            when "histogram"         {repMsg = histogramMsg(req_msg, st);}
            when "in1d"              {repMsg = in1dMsg(req_msg, st);}
            when "unique"            {repMsg = uniqueMsg(req_msg, st);}
            when "value_counts"      {repMsg = value_countsMsg(req_msg, st);}
            when "set"               {repMsg = setMsg(req_msg, st);}
            when "info"              {repMsg = infoMsg(req_msg, st);}
            when "dump"              {repMsg = dumpMsg(req_msg, st);}
            when "str"               {repMsg = strMsg(req_msg, st);}
            when "repr"              {repMsg = reprMsg(req_msg, st);}
            when "[int]"             {repMsg = intIndexMsg(req_msg, st);}
            when "[slice]"           {repMsg = sliceIndexMsg(req_msg, st);}
            when "[pdarray]"         {repMsg = pdarrayIndexMsg(req_msg, st);}
            when "[int]=val"         {repMsg = setIntIndexToValueMsg(req_msg, st);}
            when "[pdarray]=val"     {repMsg = setPdarrayIndexToValueMsg(req_msg, st);}            
            when "[pdarray]=pdarray" {repMsg = setPdarrayIndexToPdarrayMsg(req_msg, st);}            
            when "connect" {
                repMsg = "connected to arkouda server tcp://*:%t".format(ServerPort);
            }
            when "disconnect" {
                repMsg = "disconnected from arkouda server tcp://*:%t".format(ServerPort);
            }
            otherwise {
                if v {writeln("Error: unrecognized command: %s".format(req_msg)); try! stdout.flush();}
            }
        }
        
        // send responses
        // send count for now
        rep_count += 1;
        if v {writeln("repMsg:",repMsg); try! stdout.flush();}
        socket.send(repMsg);

        // end timer for command processing
        if v{writeln("<<< ", cmd," took ", getCurrentTime() - t1,"sec"); try! stdout.flush();}
    }

    writeln("requests = ",req_count," response_count = ",rep_count);
}