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

        var rep_msg: string;
        
        // peel off the command
        var fields = req_msg.split(1);
        var cmd = fields[1];
        if v {writeln(">>> ",cmd); try! stdout.flush();}
        // parse requests, execute requests, format responses
        select cmd
        {
            when "create"            {rep_msg = createMsg(req_msg, st);}
            when "delete"            {rep_msg = deleteMsg(req_msg, st);}
            when "binopvv"           {rep_msg = binopvvMsg(req_msg, st);}
            when "binopvs"           {rep_msg = binopvsMsg(req_msg, st);}
            when "binopsv"           {rep_msg = binopsvMsg(req_msg, st);}
            when "opeqvv"            {rep_msg = opeqvvMsg(req_msg, st);}
            when "opeqvs"            {rep_msg = opeqvsMsg(req_msg, st);}
            when "efunc"             {rep_msg = efuncMsg(req_msg, st);}
            when "reduction"         {rep_msg = reductionMsg(req_msg, st);}
            when "arange"            {rep_msg = arangeMsg(req_msg, st);}
            when "linspace"          {rep_msg = linspaceMsg(req_msg, st);}
            when "randint"           {rep_msg = randintMsg(req_msg, st);}
            when "histogram"         {rep_msg = histogramMsg(req_msg, st);}
            when "in1d"              {rep_msg = in1dMsg(req_msg, st);}
            when "unique"            {rep_msg = uniqueMsg(req_msg, st);}
            when "value_counts"      {rep_msg = value_countsMsg(req_msg, st);}
            when "set"               {rep_msg = setMsg(req_msg, st);}
            when "info"              {rep_msg = infoMsg(req_msg, st);}
            when "dump"              {rep_msg = dumpMsg(req_msg, st);}
            when "str"               {rep_msg = strMsg(req_msg, st);}
            when "repr"              {rep_msg = reprMsg(req_msg, st);}
            when "[int]"             {rep_msg = intIndexMsg(req_msg, st);}
            when "[slice]"           {rep_msg = sliceIndexMsg(req_msg, st);}
            when "[pdarray]"         {rep_msg = pdarrayIndexMsg(req_msg, st);}
            when "[int]=val"         {rep_msg = setIntIndexToValueMsg(req_msg, st);}
            when "[pdarray]=val"     {rep_msg = setPdarrayIndexToValueMsg(req_msg, st);}            
            when "[pdarray]=pdarray" {rep_msg = setPdarrayIndexToPdarrayMsg(req_msg, st);}            
            when "connect" {
                rep_msg = "connected to arkouda server tcp://*:%t".format(ServerPort);
            }
            when "disconnect" {
                rep_msg = "disconnected from arkouda server tcp://*:%t".format(ServerPort);
            }
            otherwise {
                if v {writeln("Error: unrecognized command: %s".format(req_msg)); try! stdout.flush();}
            }
        }
        
        // send responses
        // send count for now
        rep_count += 1;
        if v {writeln("rep_msg:",rep_msg); try! stdout.flush();}
        socket.send(rep_msg);

        // end timer for command processing
        if v{writeln("<<< ", cmd," took ", getCurrentTime() - t1,"sec"); try! stdout.flush();}
    }

    writeln("requests = ",req_count," response_count = ",rep_count);
}