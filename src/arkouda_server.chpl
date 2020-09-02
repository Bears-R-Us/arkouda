/* arkouda server
backend chapel program to mimic ndarray from numpy
This is the main driver for the arkouda server */

use FileIO;
use Security;
use ServerConfig;
use Time only;
use ZMQ only;
use Memory;
use FileSystem;
use IO;
use Path;
use MultiTypeSymbolTable;
use MultiTypeSymEntry;
use MsgProcessing;
use GenSymIO;
use SymArrayDmap;
use ServerErrorStrings;

proc initArkoudaDirectory() {
    var arkDirectory = '%s%s%s'.format(here.cwd(), pathSep,'.arkouda');
    initDirectory(arkDirectory);
    return arkDirectory;
}

proc main() {
    writeln("arkouda server version = ",arkoudaVersion); try! stdout.flush();
    writeln("memory tracking = ", memTrack); try! stdout.flush();
    const arkDirectory = initArkoudaDirectory();
    writeln("initialized the .arkouda directory %s".format(arkDirectory));

    if (memTrack) {
        writeln("getMemLimit() = ",getMemLimit());
        writeln("bytes of memoryUsed() = ",memoryUsed());
        try! stdout.flush();
    }

    var st = new owned SymTab();
    var shutdownServer = false;
    var serverToken : string;
    var serverMessage : string;

    // create and connect ZMQ socket
    var context: ZMQ.Context;
    var socket : ZMQ.Socket = context.socket(ZMQ.REP);

    // configure token authentication and server startup message accordingly
    if authenticate {
        serverToken = getArkoudaToken('%s%s%s'.format(arkDirectory, pathSep, 'tokens.txt'));
        serverMessage = "server listening on %s:%t with token %s".format(serverHostname, 
                                        ServerPort, serverToken);
    } else {
        serverMessage = "server listening on %s:%t".format(serverHostname, ServerPort);
    }

    socket.bind("tcp://*:%t".format(ServerPort));

    writeln(serverMessage); try! stdout.flush();
    
    createServerConnectionInfo();

    var reqCount: int = 0;
    var repCount: int = 0;

    var t1 = new Time.Timer();
    t1.clear();
    t1.start();

    /*
    Following processing of incoming message, sends a message back to the client.

    :arg repMsg: either a string or bytes to be sent
    */
    proc sendRepMsg(repMsg: ?t) where t==string || t==bytes {
        repCount += 1;
        if logging {
          if t==bytes {
              writeln("repMsg:"," <binary-data>");
          } else {
            writeln("repMsg:",repMsg);
          }
          try! stdout.flush();
        }
        socket.send(repMsg);
    }

    /*
    Compares the token submitted by the user with the arkouda_server token. If the
    tokens do not match, or the user did not submit a token, an ErrorWithMsg is thrown.    

    :arg token: the submitted token string
    */
    proc authenticateUser(token : string) throws {
        if token == 'None' || token.isEmpty() {
            throw new owned ErrorWithMsg("Error: access to arkouda requires a token");
        }
        else if serverToken != token {
            throw new owned ErrorWithMsg("Error: token %s does not match server token, check with server owner".format(token));
        }
    } 
   
    /*
    Parses the colon-delimted string containing the user, token, and cmd fields
    into a three-string tuple.

    :arg rawCmdSting: the colon-delimited string to be parsed
    :returns: (string,string,string)
    */ 
    proc getCommandStrings(rawCmdString : string) : (string,string,string) {
        var strings = rawCmdString.splitMsgToTuple(sep=":", numChunks=3);
        return (strings[0],strings[1],strings[2]);
    }

    /*
    Sets the shutdownServer boolean to true and sends the shutdown command to socket,
    which stops the arkouda_server listener thread and closes socket.
    */
    proc shutdown() {
        shutdownServer = true;
        repCount += 1;
        socket.send("shutdown server (%i req)".format(repCount));
    }
    
    while !shutdownServer {
        // receive message on the zmq socket
        var reqMsgRaw = socket.recv(bytes);

        reqCount += 1;

        var s0 = t1.elapsed();
        
        /*
        Separate the first tuple, which is a string binary 
        containing the message's user, token, and cmd from
        the remaining payload. Depending upon the message type 
        (string or binary) the payload is either a space-delimited
        string or bytes
        */
        const (cmdRaw, payload) = reqMsgRaw.splitMsgToTuple(2);

        var user, token, cmd: string;

        // parse requests, execute requests, format responses
        try {
            /*
            decode the string binary containing the user, token, and cmd. 
            If there is an error, discontinue processing message and send 
            an error message back to the client.
            */
            var cmdStr : string;

            try! {
                 cmdStr = cmdRaw.decode();
            } catch e: DecodeError {
               if v {
                    writeln("Error: illegal byte sequence in command: ",
                            cmdRaw.decode(decodePolicy.replace));
                    try! stdout.flush();
               }
               sendRepMsg(unknownError(e.message()));
            }

            //parse the decoded cmdString to retrieve user,token,cmd
            var (user,token,cmd) = getCommandStrings(cmdStr);

            /*
            If authentication is enabled with --authenticate flag, authenticate
            the user which for now consists of matching the submitted token
            with the token generated by the arkouda server
            */ 
            if authenticate {
                authenticateUser(token);
            }

            // If cmd is shutdown, don't bother generating a repMsg
            if cmd == "shutdown" {
                shutdown();
                if (logging) {writeln("<<< shutdown took %.17r sec".format(t1.elapsed() - s0)); 
                                                                              try! stdout.flush();}
                break;
            }

            /*
            Declare the repMsg and binaryRepMsg variables, one of which is sent to sendRepMsg
            depending upon whether a string (repMsg) or bytes (binarRepMsg) is to be returned.
            */
            var binaryRepMsg: bytes;
            var repMsg: string;

            select cmd
            {
                when "array"             {repMsg = arrayMsg(cmd, payload, st);}
                when "tondarray"         {binaryRepMsg = tondarrayMsg(cmd, payload,st);}
                when "mink"              {repMsg = minkMsg(cmd, payload, st);}
                when "maxk"              {repMsg = maxkMsg(cmd, payload, st);}
                when "intersect1d"       {repMsg = intersect1dMsg(cmd, payload, st);}
                when "setdiff1d"         {repMsg = setdiff1dMsg(cmd, payload, st);}
                when "setxor1d"          {repMsg = setxor1dMsg(cmd, payload, st);}
                when "union1d"           {repMsg = union1dMsg(cmd, payload, st);}
                when "segmentLengths"    {repMsg = segmentLengthsMsg(cmd, payload, st);}
                when "segmentedHash"     {repMsg = segmentedHashMsg(cmd, payload, st);}
                when "segmentedEfunc"    {repMsg = segmentedEfuncMsg(cmd, payload, st);}
                when "segmentedIndex"    {repMsg = segmentedIndexMsg(cmd, payload, st);}
                when "segmentedBinopvv"  {repMsg = segBinopvvMsg(cmd, payload, st);}
                when "segmentedBinopvs"  {repMsg = segBinopvsMsg(cmd, payload, st);}
                when "segmentedGroup"    {repMsg = segGroupMsg(cmd, payload, st);}
                when "segmentedIn1d"     {repMsg = segIn1dMsg(cmd, payload, st);}
                when "lshdf"             {repMsg = lshdfMsg(cmd, payload, st);}
                when "readhdf"           {repMsg = readhdfMsg(cmd, payload, st);}
                when "readAllHdf"        {repMsg = readAllHdfMsg(cmd, payload, st);}
                when "tohdf"             {repMsg = tohdfMsg(cmd, payload, st);}
                when "create"            {repMsg = createMsg(cmd, payload, st);}
                when "delete"            {repMsg = deleteMsg(cmd, payload, st);}
                when "binopvv"           {repMsg = binopvvMsg(cmd, payload, st);}
                when "binopvs"           {repMsg = binopvsMsg(cmd, payload, st);}
                when "binopsv"           {repMsg = binopsvMsg(cmd, payload, st);}
                when "opeqvv"            {repMsg = opeqvvMsg(cmd, payload, st);}
                when "opeqvs"            {repMsg = opeqvsMsg(cmd, payload, st);}
                when "efunc"             {repMsg = efuncMsg(cmd, payload, st);}
                when "efunc3vv"          {repMsg = efunc3vvMsg(cmd, payload, st);}
                when "efunc3vs"          {repMsg = efunc3vsMsg(cmd, payload, st);}
                when "efunc3sv"          {repMsg = efunc3svMsg(cmd, payload, st);}
                when "efunc3ss"          {repMsg = efunc3ssMsg(cmd, payload, st);}
                when "reduction"         {repMsg = reductionMsg(cmd, payload, st);}
                when "countReduction"    {repMsg = countReductionMsg(cmd, payload, st);}
                when "countLocalRdx"     {repMsg = countLocalRdxMsg(cmd, payload, st);}
                when "findSegments"      {repMsg = findSegmentsMsg(cmd, payload, st);}
                when "findLocalSegments" {repMsg = findLocalSegmentsMsg(cmd, payload, st);}
                when "segmentedReduction"{repMsg = segmentedReductionMsg(cmd, payload, st);}
                when "segmentedLocalRdx" {repMsg = segmentedLocalRdxMsg(cmd, payload, st);}
                when "arange"            {repMsg = arangeMsg(cmd, payload, st);}
                when "linspace"          {repMsg = linspaceMsg(cmd, payload, st);}
                when "randint"           {repMsg = randintMsg(cmd, payload, st);}
                when "randomNormal"      {repMsg = randomNormalMsg(cmd, payload, st);}
                when "randomStrings"     {repMsg = randomStringsMsg(cmd, payload, st);}
                when "histogram"         {repMsg = histogramMsg(cmd, payload, st);}
                when "in1d"              {repMsg = in1dMsg(cmd, payload, st);}
                when "unique"            {repMsg = uniqueMsg(cmd, payload, st);}
                when "value_counts"      {repMsg = value_countsMsg(cmd, payload, st);}
                when "set"               {repMsg = setMsg(cmd, payload, st);}
                when "info"              {repMsg = infoMsg(cmd, payload, st);}
                when "str"               {repMsg = strMsg(cmd, payload, st);}
                when "repr"              {repMsg = reprMsg(cmd, payload, st);}
                when "[int]"             {repMsg = intIndexMsg(cmd, payload, st);}
                when "[slice]"           {repMsg = sliceIndexMsg(cmd, payload, st);}
                when "[pdarray]"         {repMsg = pdarrayIndexMsg(cmd, payload, st);}
                when "[int]=val"         {repMsg = setIntIndexToValueMsg(cmd, payload, st);}
                when "[pdarray]=val"     {repMsg = setPdarrayIndexToValueMsg(cmd, payload, st);}
                when "[pdarray]=pdarray" {repMsg = setPdarrayIndexToPdarrayMsg(cmd, payload, st);}
                when "[slice]=val"       {repMsg = setSliceIndexToValueMsg(cmd, payload, st);}
                when "[slice]=pdarray"   {repMsg = setSliceIndexToPdarrayMsg(cmd, payload, st);}
                when "argsort"           {repMsg = argsortMsg(cmd, payload, st);}
                when "coargsort"         {repMsg = coargsortMsg(cmd, payload, st);}
                when "concatenate"       {repMsg = concatenateMsg(cmd, payload, st);}
                when "localArgsort"      {repMsg = localArgsortMsg(cmd, payload, st);}
                when "sort"              {repMsg = sortMsg(cmd, payload, st);}
                when "joinEqWithDT"      {repMsg = joinEqWithDTMsg(cmd, payload, st);}
                when "getconfig"         {repMsg = getconfigMsg(cmd, payload, st);}
                when "getmemused"        {repMsg = getmemusedMsg(cmd, payload, st);}
                when "register"          {repMsg = registerMsg(cmd, payload, st);}
                when "attach"            {repMsg = attachMsg(cmd, payload, st);}
                when "unregister"        {repMsg = unregisterMsg(cmd, payload, st);}
                when "connect" {
                    if authenticate {
                        repMsg = "connected to arkouda server tcp://*:%t as user %s with token %s".format(ServerPort,user,token);
                    } else {
                        repMsg = "connected to arkouda server tcp://*:%t".format(ServerPort);
                    }
                }
                when "disconnect" {
                    repMsg = "disconnected from arkouda server tcp://*:%t".format(ServerPort);
                }
                when "noop" {
                    repMsg = "noop";
                    if v { writeln("no-op"); try! stdout.flush(); }
                }
                when "ruok" {
                    repMsg = "imok";
                }
                otherwise {
                    repMsg = "Error: unrecognized command: %s".format(cmd);
                }
            }

            //Determine if a string (repMsg) or binary (binaryRepMsg) is to be returned and send response to client           
            if repMsg.isEmpty() {
                sendRepMsg(binaryRepMsg);
            } else {
                sendRepMsg(repMsg);
            }

            // log that the request message has been handled and reply message has been sent along with time to do so
            if (logging) {writeln("<<< %s took %.17r sec".format(cmd, t1.elapsed() - s0)); try! stdout.flush();}
            if (logging && memTrack) {writeln("bytes of memory used after command = ", 
                                                               memoryUsed():uint * numLocales:uint); try! stdout.flush();}
        } catch (e: ErrorWithMsg) {
            sendRepMsg(e.msg);
            if (logging) {writeln("<<< %s resulted in error %s in  %.17r sec".format(cmd, 
                                                                              e.msg, t1.elapsed() - s0)); try! stdout.flush();}
        } catch (e: Error) {
            sendRepMsg(unknownError(e.message()));
            if (logging) {writeln("<<< %s resulted in error: %s in %.17r sec".format(cmd, e.message(),
                                                                                     t1.elapsed() - s0)); try! stdout.flush();}
        }
    }

    t1.stop();

    deleteServerConnectionInfo();

    writeln("requests = ",reqCount," responseCount = ",repCount," elapsed sec = ",t1.elapsed());
}

/*
Creates the serverConnectionInfo file on arkouda_server startup
*/
proc createServerConnectionInfo() {
    use IO;
    if !serverConnectionInfo.isEmpty() {
        try! {
            var w = open(serverConnectionInfo, iomode.cw).writer();
            w.writef("%s %t\n", serverHostname, ServerPort);
        }
    }
}

/*
Deletes the serverConnetionFile on arkouda_server shutdown
*/
proc deleteServerConnectionInfo() {
    use FileSystem;
    if !serverConnectionInfo.isEmpty() {
        try! {
            remove(serverConnectionInfo);
        }
    }
}
