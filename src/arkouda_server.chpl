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
use Logging;
use Path;
use MultiTypeSymbolTable;
use MultiTypeSymEntry;
use MsgProcessing;
use GenSymIO;
use Reflection;
use SymArrayDmap;
use ServerErrorStrings;
use Message;

const asLogger = new Logger();

enum MsgType {REGULAR,WARNING,ERROR}
enum MsgFormat {STRING,BINARY}

class ReplyMsg {
    var msg: string;
    var msgType: MsgType;
    var msgFormat: MsgFormat;
}

if v {
    asLogger.level = LogLevel.DEBUG;
} else {
    asLogger.level = LogLevel.INFO;
}

proc initArkoudaDirectory() {
    var arkDirectory = '%s%s%s'.format(here.cwd(), pathSep,'.arkouda');
    initDirectory(arkDirectory);
    return arkDirectory;
}

proc main() {
    asLogger.info(getModuleName(), getRoutineName(), getLineNumber(),
                                               "arkouda server version = %s".format(arkoudaVersion));
    asLogger.info(getModuleName(), getRoutineName(), getLineNumber(),
                                               "memory tracking = %t".format(memTrack));
    const arkDirectory = initArkoudaDirectory();
    asLogger.info(getModuleName(), getRoutineName(), getLineNumber(),
                                       "initialized the .arkouda directory %s".format(arkDirectory));

    if (memTrack) {
        asLogger.info(getModuleName(), getRoutineName(), getLineNumber(), 
                                               "getMemLimit() %i".format(getMemLimit()));
        asLogger.info(getModuleName(), getRoutineName(), getLineNumber(), 
                                               "bytes of memoryUsed() = %i".format(memoryUsed()));
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
        serverMessage = ">>>>>>>>>>>>>>> server listening on tcp://%s:%t?token=%s " +
                        "<<<<<<<<<<<<<<<".format(serverHostname, ServerPort, serverToken);
    } else {
        serverMessage = ">>>>>>>>>>>>>>> server listening on tcp://%s:%t <<<<<<<<<<<<<<<".format(
                                        serverHostname, ServerPort);
    }

    socket.bind("tcp://*:%t".format(ServerPort));

    const boundary = "**************************************************************************" +
                   "**************************";

    asLogger.info(getModuleName(), getRoutineName(), getLineNumber(), boundary);
    asLogger.info(getModuleName(), getRoutineName(), getLineNumber(), serverMessage);
    asLogger.info(getModuleName(), getRoutineName(), getLineNumber(), boundary);
    
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
              asLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "repMsg: <binary-data>");
          } else {
              asLogger.info(getModuleName(),getRoutineName(),getLineNumber(), 
                                                        "repMsg: %s".format(repMsg));
          }
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
        asLogger.info(getModuleName(), getRoutineName(), getLineNumber(),"stuff %t".format(strings));      
        return (strings[0],strings[1],strings[2]);
    }

    proc extractCommand(cmdString : string) : CommandMsg throws {
        var cm = new CommandMsg();
        deserialize(cm,cmdString);
        return cm;
    }
    
    /*
    Sets the shutdownServer boolean to true and sends the shutdown command to socket,
    which stops the arkouda_server listener thread and closes socket.
    */
    proc shutdown() {
        shutdownServer = true;
        repCount += 1;
        socket.send(generateJsonReplyMsg(msg="shutdown server (%i req)".format(repCount), 
                         msgType=MsgType.REGULAR,msgFormat=MsgFormat.STRING));
    }
    
    while !shutdownServer {
        // receive message on the zmq socket
        var reqMsgRaw = socket.recv(bytes);

        reqCount += 1;

        var s0 = t1.elapsed();
        
        /*
         * Separate the first tuple, which is a string binary 
         * containing the JSON binary string encapsulating user, token, cmd, and args from
         * the remaining payload. Depending upon
         */
        var (cmdRaw, payload) = reqMsgRaw.splitMsgToTuple(b"BINARY_PAYLOAD",2);
        var user, token, cmd: string;

        // parse requests, execute requests, format responses
        try {
            /*
             * Decode the string binary containing the user, token, and cmd. 
             * If there is an error, discontinue processing message and send 
             * an error message back to the client.
             */
            var cmdStr : string;

            try! {
                cmdStr = cmdRaw.decode();
            } catch e: DecodeError {
                asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                       "illegal byte sequence in command: %t".format(cmdRaw.decode(decodePolicy.replace)));
                sendRepMsg(generateJsonReplyMsg(msg=unknownError(e.message()),msgType=MsgType.ERROR,
                                                 msgFormat=MsgFormat.STRING));
            }

            //parse the decoded cmdString to retrieve user,token,cmd
            asLogger.info(getModuleName(),getRoutineName(),getLineNumber(),"INCOMING CMD %s".format(cmdStr));
            var msg    = extractCommand(cmdStr);
            var user   = msg.user;
            var token  = msg.token;
            var cmd    = msg.cmd;
            var format = msg.format;
            var args   = msg.args;

            /*
             * If authentication is enabled with --authenticate flag, authenticate
             * the user which for now consists of matching the submitted token
             * with the token generated by the arkouda server
             */ 
            if authenticate {
                authenticateUser(token);
            }

            if (logging) {
              try {
                if (cmd != "array") {
                  asLogger.info(getModuleName(), getRoutineName(), getLineNumber(),
                                                     ">>> %t %t".format(cmd, 
                                                    payload.decode(decodePolicy.replace)));
                } else {
                  asLogger.info(getModuleName(), getRoutineName(), getLineNumber(),
                                                     ">>> %s [binary data]".format(cmd));
                }
              } catch {
                // No action on error
              }
            }

            // If cmd is shutdown, don't bother generating a repMsg
            if cmd == "shutdown" {
                shutdown();
                if (logging) {
                    asLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                         "<<< shutdown took %.17r sec".format(t1.elapsed() - s0));
                }
                break;
            }

            /*
             * Declare the repMsg and binaryRepMsg variables, one of which is sent to sendRepMsg
             * depending upon whether a string (repMsg) or bytes (binaryRepMsg) is to be returned.
             */
            var binaryRepMsg: bytes;
            var repMsg: string;

            asLogger.info(getModuleName(),getRoutineName(),getLineNumber(),"cmd: %t payload: %t".format(cmd,payload));
            select cmd
            {
                when "array"             {repMsg = arrayMsg(cmd, payload, st);}
                when "tondarray"         {binaryRepMsg = tondarrayMsg(cmd, args, st);}
                when "cast"              {repMsg = castMsg(cmd, args, st);}
                when "mink"              {repMsg = minkMsg(cmd, args, st);}
                when "maxk"              {repMsg = maxkMsg(cmd, args, st);}
                when "intersect1d"       {repMsg = intersect1dMsg(cmd, args, st);}
                when "setdiff1d"         {repMsg = setdiff1dMsg(cmd, args, st);}
                when "setxor1d"          {repMsg = setxor1dMsg(cmd, args, st);}
                when "union1d"           {repMsg = union1dMsg(cmd, args, st);}
                when "segmentLengths"    {repMsg = segmentLengthsMsg(cmd, args, st);}
                when "segmentedHash"     {repMsg = segmentedHashMsg(cmd, args, st);}
                when "segmentedEfunc"    {repMsg = segmentedEfuncMsg(cmd, args, st);}
                when "segmentedPeel"     {repMsg = segmentedPeelMsg(cmd, args, st);}
                when "segmentedIndex"    {repMsg = segmentedIndexMsg(cmd, args, st);}
                when "segmentedBinopvv"  {repMsg = segBinopvvMsg(cmd, args, st);}
                when "segmentedBinopvs"  {repMsg = segBinopvsMsg(cmd, args, st);}
                when "segmentedGroup"    {repMsg = segGroupMsg(cmd, args, st);}
                when "segmentedIn1d"     {repMsg = segIn1dMsg(cmd, args, st);}
                when "lshdf"             {repMsg = lshdfMsg(cmd, args, st);}
                when "readhdf"           {repMsg = readhdfMsg(cmd, args, st);}
                when "readAllHdf"        {repMsg = readAllHdfMsg(cmd, args, st);}
                when "tohdf"             {repMsg = tohdfMsg(cmd, args, st);}
                when "create"            {repMsg = createMsg(cmd, args, st);}
                when "delete"            {repMsg = deleteMsg(cmd, args, st);}
                when "binopvv"           {repMsg = binopvvMsg(cmd, args, st);}
                when "binopvs"           {repMsg = binopvsMsg(cmd, args, st);}
                when "binopsv"           {repMsg = binopsvMsg(cmd, args, st);}
                when "opeqvv"            {repMsg = opeqvvMsg(cmd, args, st);}
                when "opeqvs"            {repMsg = opeqvsMsg(cmd, args, st);}
                when "efunc"             {repMsg = efuncMsg(cmd, args, st);}
                when "efunc3vv"          {repMsg = efunc3vvMsg(cmd, args, st);}
                when "efunc3vs"          {repMsg = efunc3vsMsg(cmd, args, st);}
                when "efunc3sv"          {repMsg = efunc3svMsg(cmd, args, st);}
                when "efunc3ss"          {repMsg = efunc3ssMsg(cmd, args, st);}
                when "reduction"         {repMsg = reductionMsg(cmd, args, st);}
                when "countReduction"    {repMsg = countReductionMsg(cmd, args, st);}
                when "findSegments"      {repMsg = findSegmentsMsg(cmd, args, st);}
                when "segmentedReduction"{repMsg = segmentedReductionMsg(cmd, args, st);}
                when "broadcast"         {repMsg = broadcastMsg(cmd, args, st);}
                when "arange"            {repMsg = arangeMsg(cmd, args, st);}
                when "linspace"          {repMsg = linspaceMsg(cmd, args, st);}
                when "randint"           {repMsg = randintMsg(cmd, args, st);}
                when "randomNormal"      {repMsg = randomNormalMsg(cmd, args, st);}
                when "randomStrings"     {repMsg = randomStringsMsg(cmd, args, st);}
                when "histogram"         {repMsg = histogramMsg(cmd, args, st);}
                when "in1d"              {repMsg = in1dMsg(cmd, args, st);}
                when "unique"            {repMsg = uniqueMsg(cmd, args, st);}
                when "value_counts"      {repMsg = value_countsMsg(cmd, args, st);}
                when "set"               {repMsg = setMsg(cmd, args, st);}
                when "info"              {repMsg = infoMsg(cmd, args, st);}
                when "str"               {repMsg = strMsg(cmd, args, st);}
                when "repr"              {repMsg = reprMsg(cmd, args, st);}
                when "[int]"             {repMsg = intIndexMsg(cmd, args, st);}
                when "[slice]"           {repMsg = sliceIndexMsg(cmd, args, st);}
                when "[pdarray]"         {repMsg = pdarrayIndexMsg(cmd, args, st);}
                when "[int]=val"         {repMsg = setIntIndexToValueMsg(cmd, args, st);}
                when "[pdarray]=val"     {repMsg = setPdarrayIndexToValueMsg(cmd, args, st);}
                when "[pdarray]=pdarray" {repMsg = setPdarrayIndexToPdarrayMsg(cmd, args, st);}
                when "[slice]=val"       {repMsg = setSliceIndexToValueMsg(cmd, args, st);}
                when "[slice]=pdarray"   {repMsg = setSliceIndexToPdarrayMsg(cmd, args, st);}
                when "argsort"           {repMsg = argsortMsg(cmd, args, st);}
                when "coargsort"         {repMsg = coargsortMsg(cmd, args, st);}
                when "concatenate"       {repMsg = concatenateMsg(cmd, args, st);}
                when "sort"              {repMsg = sortMsg(cmd, args, st);}
                when "joinEqWithDT"      {repMsg = joinEqWithDTMsg(cmd, args, st);}
                when "getconfig"         {repMsg = getconfigMsg(cmd, args, st);}
                when "getmemused"        {repMsg = getmemusedMsg(cmd, args, st);}
                when "register"          {repMsg = registerMsg(cmd, args, st);}
                when "attach"            {repMsg = attachMsg(cmd, args, st);}
                when "unregister"        {repMsg = unregisterMsg(cmd, args, st);}
                when "clear"             {repMsg = clearMsg(cmd, args, st);}               
                when "connect" {
                    if authenticate {
                        repMsg = "connected to arkouda server tcp://*:%i as user %s with token %s".format(
                                                          ServerPort,user,token);
                    } else {
                        repMsg = "connected to arkouda server tcp://*:%i".format(ServerPort);
                    }                 
                }
                when "disconnect" {      
                    repMsg = "disconnected from arkouda server tcp://*:%i".format(ServerPort);
                }
                when "noop" {
                    repMsg = "noop";
                    asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"no-op");
                }
                when "ruok" {
                    repMsg = "imok";
                }
                otherwise {
                    repMsg = "Error: unrecognized command: %s".format(cmd);
                    asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                }
            }

            //Determine if a string (repMsg) or binary (binaryRepMsg) is to be returned and send response           
            if repMsg.isEmpty() {
                sendRepMsg(binaryRepMsg);
            } else {
                var msgType: MsgType;
                
                if repMsg.find('Error') > -1 {
                    msgType = MsgType.ERROR;
                } else if repMsg.find('Warning') > -1 {
                    msgType = MsgType.WARNING;
                } else {
                    msgType = MsgType.REGULAR;
                }
                sendRepMsg(generateJsonReplyMsg(msg=repMsg,msgType=msgType,msgFormat=MsgFormat.STRING));
            }

            /*
             * log that the request message has been handled and reply message has been sent along with 
             * the time to do so
             */
            if logging {
                asLogger.info(getModuleName(),getRoutineName(),getLineNumber(), 
                                                  "<<< %s took %.17r sec".format(cmd, t1.elapsed() - s0));
            }
            if (logging && memTrack) {
                asLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                       "bytes of memory used after command %t".format(memoryUsed():uint * numLocales:uint));
            }
        } catch (e: ErrorWithMsg) {
            sendRepMsg(generateJsonReplyMsg(msg=e.msg,msgType=MsgType.ERROR, msgFormat=MsgFormat.STRING));
            if logging {
                asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                    "<<< %s resulted in error %s in  %.17r sec".format(cmd, e.msg, t1.elapsed() - s0));
            }
        } catch (e: Error) {
            sendRepMsg(generateJsonReplyMsg(msg=unknownError(e.message()),msgType=MsgType.ERROR, 
                                                         msgFormat=MsgFormat.STRING));
            if logging {
                asLogger.error(getModuleName(), getRoutineName(), getLineNumber(), 
                    "<<< %s resulted in error: %s in %.17r sec".format(cmd, e.message(),t1.elapsed() - s0));
            }
        }
    }

    t1.stop();

    deleteServerConnectionInfo();

    asLogger.info(getModuleName(), getRoutineName(), getLineNumber(),
               "requests = %i responseCount = %i elapsed sec = %i".format(reqCount,repCount,t1.elapsed()));
}

/*
 * Generates JSON-formatted reply message
 */
proc generateJsonReplyMsg(msg: string, msgType: MsgType, msgFormat: MsgFormat) {
    return "%jt".format(new ReplyMsg(msg=msg,msgType=msgType, msgFormat=msgFormat));
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
