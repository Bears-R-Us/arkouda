module ServerDaemon {
    use FileIO;
    use IO;
    use Security;
    use ServerConfig;
    use ServerErrors;
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
    use CommandMap;
    use Errors;
    use List;
    use ExternalIntegration;
    use MetricsMsg;

    enum ServerDaemonType {DEFAULT,INTEGRATION,METRICS}

    private config const logLevel = ServerConfig.logLevel;
    const sdLogger = new Logger(logLevel);

    private config const daemonTypes = 'ServerDaemonType.DEFAULT';
    
    var serverDaemonTypes = try! getDaemonTypes();

    /**
     * Retrieves a list of 1..n ServerDaemonType objects generated 
     * from the comma-delimited list of ServerDaemonType strings
     * provided in the daemonTypes command-line parameter
     */
    proc getDaemonTypes() throws {
        var types = new list(ServerDaemonType);
        var rawTypes = daemonTypes.split(',');

        for rt in rawTypes {
            var daemonType: ServerDaemonType;
            daemonType = rt: ServerDaemonType;
            types.append(daemonType);
        }
        return types;
    }    

    /**
     * Returns a boolean indicating if Arkouda is configured to generate and
     * make available metrics via a dedicated metrics socket
     */
    proc metricsEnabled() {
        return serverDaemonTypes.contains(ServerDaemonType.METRICS);
    }

    /**
     * Returns a boolean indicating whether Arkouda is configured to 
     * register/deregister with an external system such as Kubernetes.
     */
    proc integrationEnabled() {
        return serverDaemonTypes.contains(ServerDaemonType.INTEGRATION);
    }
    
    /**
     * Returns a boolean indicating if there are multiple ServerDaemons
     */
    proc multipleServerDaemons() {
        return serverDaemonTypes.size > 1;
    }

    /**
     * Generates the Kubernetes app name for Arkouda, which varies based
     * upon which pod corresponds to Locale 0.
     */
    proc register(endpoint: ServiceEndpoint) throws {
        on Locales[0] {
            var appName: string;

            if serverHostname.count('arkouda-locale') > 0 {
                appName = 'arkouda-locale';
            } else {
                appName = 'arkouda-server';
            }
            registerWithExternalSystem(appName, endpoint);
        }
    }

    /**
     * The ArkoudaServerDaemon class defines the run and shutdown 
     * functions all derived classes must override
     */
    class ArkoudaServerDaemon {
        var st = new owned SymTab();
        var shutdownDaemon = false;
        var port: int;
 
        proc run() throws {
            throw new NotImplementedError("run() must be overridden",
                                          getLineNumber(),
                                          getRoutineName(),
                                          getModuleName());
        }
        
        /**
         * Prompts the ArkoudaServerDaemon to initiate a shutdown, triggering
         * a change in state which will cause the derived ArkoudaServerDaemon
         * to (1) exit the daemon loop within the run function and (2) execute 
         * the shutdown function.
         */
        proc requestShutdown(user: string) throws {
            this.shutdownDaemon = true;
        }
        
        /*
         * Converts the incoming request JSON string into RequestMsg object.
         */
        proc extractRequest(request : string) : RequestMsg throws {
            var rm = new RequestMsg();
            deserialize(rm, request);
            return rm;
        }
        
        /**
         * Encapsulates logic that is to be invoked once a ArkoduaSeverDaemon
         * has exited the daemon loop.
         */
        proc shutdown() throws {
            sdLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                              "shutdown sequence complete");   
        }
    }

    /**
     * The DefaultServerDaemon class serves as the default Arkouda server
     * daemon which is run within the arkouda_server driver
     */
    class DefaultServerDaemon : ArkoudaServerDaemon {
        var serverToken : string;
        var arkDirectory : string;
        var connectUrl : string;
        var reqCount: int = 0;
        var repCount: int = 0;
        
        var context: ZMQ.Context;
        var socket : ZMQ.Socket;        
       
        proc init() {
            this.socket = this.context.socket(ZMQ.REP); 
            try! this.socket.bind("tcp://*:%t".format(ServerPort));
        }
        
        proc getConnectUrl(token: string) throws {
            if token.isEmpty() {
                return "tcp://%s:%t".format(serverHostname, 
                                            ServerPort);
            } else {
                return "tcp://%s:%i?token=%s".format(serverHostname,
                                                     ServerPort,
                                                     token);
            }
        }

        proc printServerSplashMessage(token: string, arkDirectory: string) throws {
            var verMessage = "arkouda server version = %s".format(arkoudaVersion);
            var chplVerMessage = "built with chapel version%s".format(chplVersion);
            var dirMessage = ".arkouda directory %s".format(arkDirectory);
            var memLimMessage =  "memory limit = %i".format(getMemLimit());
            var memUsedMessage = "bytes of memory used = %i".format(getMemUsed());
            var serverMessage: string;
    
            const buff = '                         ';
    
            proc adjustMsg(msg: string) throws {
                if msg.size % 2 != 0 {
                    return msg + ' ';
                } else {
                    return msg;
                }   
            }
    
            proc generateBuffer(longSegment: string, shortSegment: string) : string {
                var buffSize = (longSegment.size - shortSegment.size)/2 - 2;
                var buffer: string;
                var counter = 0;
        
                while counter <= buffSize {
                    buffer+=' ';
                    counter+=1;
                }           
                return buffer;
            }

            serverMessage = "server listening on %s".format(this.connectUrl);
            serverMessage = adjustMsg(serverMessage);      
            serverMessage = "%s %s %s".format(buff,serverMessage,buff);
        
            var vBuff = generateBuffer(serverMessage,verMessage);
            verMessage = adjustMsg(verMessage);
            verMessage = "*%s %s %s*".format(vBuff,verMessage,vBuff);
            
            var cvBuff = generateBuffer(serverMessage,chplVerMessage);
            chplVerMessage = adjustMsg(chplVerMessage);
            chplVerMessage = "*%s %s %s*".format(cvBuff,chplVerMessage,cvBuff);

            var mlBuff = generateBuffer(serverMessage,memLimMessage);
            memLimMessage = adjustMsg(memLimMessage);
            memLimMessage = "*%s %s %s*".format(mlBuff,memLimMessage,mlBuff);

            var muBuff = generateBuffer(serverMessage,memUsedMessage);
            memUsedMessage = adjustMsg(memUsedMessage);
            memUsedMessage = "*%s %s %s*".format(muBuff,memUsedMessage,muBuff);
        
            var blankBuffer: string;
            var counter = 0;
        
            while counter < serverMessage.size {
                blankBuffer+=' ';
                counter+=1;
            }

            var blankLine = '*%s*'.format(blankBuffer);
        
            var tag = '*';
            counter = 0;
        
            while counter <= serverMessage.size {
                tag+='*';
                counter+=1;
            }

            writeln();
            writeln();
            writeln(tag);
            writeln(tag);
            writeln(blankLine);
            writeln('*%s*'.format(serverMessage));
            writeln(verMessage);
            writeln(chplVerMessage);

            if (memTrack) {
                writeln(memLimMessage);
                writeln(memUsedMessage);
            }

            writeln(blankLine);
            writeln(tag);
            writeln(tag);
            writeln();
            writeln();
            stdout.flush();
        }

        /*
        Creates the serverConnectionInfo file on arkouda_server startup
        */
        proc createServerConnectionInfo() throws {
            use IO;
            if !serverConnectionInfo.isEmpty() {
                sdLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                               'writing serverConnectionInfo to %t'.format(serverConnectionInfo));
                try! {
                    var w = open(serverConnectionInfo, iomode.cw).writer();
                    w.writef("%s %t %s\n",serverHostname,ServerPort,this.connectUrl);
                }
            }
        }

        /*
        Deletes the serverConnetionFile on arkouda_server shutdown
        */
        proc deleteServerConnectionInfo() throws {
            use FileSystem;
            try {
                if !serverConnectionInfo.isEmpty() {
                    remove(serverConnectionInfo);
                }
            } catch fnfe : FileNotFoundError {
               sdLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                              "The serverConnectionInfo file was not found %s".format(fnfe.message()));
            } catch e : Error {
               sdLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                              "Error in deleting serverConnectionInfo file %s".format(e.message()));    
            }
        }
        
        /*
        Following processing of incoming message, sends a message back to the client.

        :arg repMsg: either a string or bytes to be sent
        */
        proc sendRepMsg(repMsg: ?t) throws where t==string || t==bytes {
            this.repCount += 1;
            if trace {
                if t==bytes {
                    sdLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "repMsg: <binary-data>");
                } else {
                 sdLogger.info(getModuleName(),getRoutineName(),getLineNumber(), 
                                                        "repMsg: %s".format(repMsg));
                }
            }
            this.socket.send(repMsg);
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
        Sets the shutdownDaemon boolean to true and sends the shutdown command to socket,
        which stops the arkouda_server listener thread and closes socket.
        */
        override proc requestShutdown(user: string) throws {
            if saveUsedModules then
                writeUsedModules();
            super.requestShutdown(user);
            this.repCount += 1;
            this.socket.send(serialize(msg="shutdown server (%i req)".format(repCount), 
                         msgType=MsgType.NORMAL,msgFormat=MsgFormat.STRING, user=user));
        }

        /**
         * Register our server commands in the CommandMap
         * There are 3 general types
         * 1. Standard, required commands which adhere to the standard Message signature
         * 2. Specialized, required commands which do not adhere to the standard Message signature
         * 3. "Optional" modules which are included at compilation time via ServerModules.cfg
         */
        proc registerServerCommands() {
            registerBinaryFunction("tondarray", tondarrayMsg);
            registerFunction("create", createMsg);
            registerFunction("delete", deleteMsg);
            registerFunction("set", setMsg);
            registerFunction("info", infoMsg);
            registerFunction("str", strMsg);
            registerFunction("repr", reprMsg);
            registerFunction("getconfig", getconfigMsg);
            registerFunction("getmemused", getmemusedMsg);
            registerFunction("getCmdMap", getCommandMapMsg);
            registerFunction("clear", clearMsg);
            registerFunction("lsany", lsAnyMsg);
            registerFunction("readany", readAnyMsg);
            registerFunction("getfiletype", getFileTypeMsg);

            // For a few specialized cmds we're going to add dummy functions, so they
            // get added to the client listing of available commands. They will be
            // intercepted in the cmd processing select statement and processed specially
            registerFunction("array", akMsgSign);
            registerFunction("connect", akMsgSign);
            registerFunction("disconnect", akMsgSign);
            registerFunction("noop", akMsgSign);
            registerFunction("ruok", akMsgSign);
            registerFunction("shutdown", akMsgSign);
        }
        
        proc initArkoudaDirectory() throws {
            var arkDirectory = '%s%s%s'.format(here.cwd(), pathSep,'.arkouda');
            initDirectory(arkDirectory);
            return arkDirectory;
        }


        /**
         * The overridden shutdown function calls exit(0) if there are multiple 
         * ServerDaemons configured for this Arkouda instance.
         */
        override proc shutdown() throws {        
            if multipleServerDaemons() {
                sdLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "multiple ServerDaemons, invoking exit(0)");  
                exit(0);
            }
        }

        override proc run() throws {
            this.arkDirectory = this.initArkoudaDirectory();

            if authenticate {
                this.serverToken = getArkoudaToken('%s%s%s'.format(this.arkDirectory, pathSep, 'tokens.txt'));
            }

            sdLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                               "initialized the .arkouda directory %s".format(this.arkDirectory));
    
            this.connectUrl = this.getConnectUrl(this.serverToken);
            this.createServerConnectionInfo();
            if serverInfoNoSplash {
                writeln(getConfig());
                stdout.flush();
            } else {
                this.printServerSplashMessage(this.serverToken,this.arkDirectory);
            }
            this.registerServerCommands();
                    
            var t1 = new Time.Timer();
            t1.clear();
            t1.start();            
        
            while !this.shutdownDaemon {
                // receive message on the zmq socket
                var reqMsgRaw = socket.recv(bytes);

                this.reqCount += 1;

                var s0 = t1.elapsed();
        
                /*
                 * Separate the first tuple, which is a string binary containing the JSON binary
                 * string encapsulating user, token, cmd, message format and args from the 
                 * remaining payload.
                 */
                var (rawRequest, _) = reqMsgRaw.splitMsgToTuple(b"BINARY_PAYLOAD",2);
                var payload = if reqMsgRaw.endsWith(b"BINARY_PAYLOAD") then socket.recv(bytes) else b"";
                var user, token, cmd: string;

                // parse requests, execute requests, format responses
                try {
                    /*
                     * Decode the string binary containing the JSON-formatted request string. 
                     * If there is an error, discontinue processing message and send an error
                     * message back to the client.
                     */
                    var request : string;

                    try! {
                        request = rawRequest.decode();
                    } catch e: DecodeError {
                        sdLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                           "illegal byte sequence in command: %t".format(
                                          rawRequest.decode(decodePolicy.replace)));
                        sendRepMsg(serialize(msg=unknownError(e.message()),msgType=MsgType.ERROR,
                                                 msgFormat=MsgFormat.STRING, user="Unknown"));
                    }

                    // deserialize the decoded, JSON-formatted cmdStr into a RequestMsg
                    var msg: RequestMsg  = extractRequest(request);
                    user   = msg.user;
                    token  = msg.token;
                    cmd    = msg.cmd;
                    var format = msg.format;
                    var args   = msg.args;
                    var size: int;
                    try {
                            size = msg.size: int;
                    }
                    catch e {
                        sdLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                               "Argument List size is not an integer. %s cannot be cast".format(msg.size));
                        sendRepMsg(serialize(msg=unknownError(e.message()),msgType=MsgType.ERROR,
                                                         msgFormat=MsgFormat.STRING, user="Unknown"));
                    }

                    var msgArgs: owned MessageArgs;
                    if size > 0 {
                        msgArgs = parseMessageArgs(args, size);
                    }
                    else {
                        msgArgs = new owned MessageArgs();
                    }

                    /*
                     * If authentication is enabled with the --authenticate flag, authenticate
                     * the user which for now consists of matching the submitted token
                     * with the token generated by the arkouda server
                     */ 
                    if authenticate {
                        authenticateUser(token);
                    }

                    if (trace) {
                        try {
                            if (cmd != "array") {
                                sdLogger.info(getModuleName(), getRoutineName(), getLineNumber(),
                                                     ">>> %t %t".format(cmd, args));
                            } else {
                                sdLogger.info(getModuleName(), getRoutineName(), getLineNumber(),
                                                     ">>> %s [binary data]".format(cmd));
                            }
                       } catch {
                           // No action on error
                       }
                }

                inline proc sendShutdownRequest(user: string) throws {
                    requestShutdown(user=user);
                    if (trace) {
                        sdLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                        "<<< shutdown initiated by %s took %.17r sec".format(user, 
                                                t1.elapsed() - s0));
                    }
                }

                // If cmd is shutdown, don't bother generating a repMsg
                if cmd == "shutdown" {
                    sendShutdownRequest(user=user);
                    break;
                }

                /*
                 * For messages that return a string repTuple is filled. For binary
                 * messages the message is sent directly to minimize copies.
                 */
                var repTuple: MsgTuple;
            
                /**
                 * Command processing: Look for our specialized, default commands first, then check the command maps
                 * Note: Our specialized commands have been added to the commandMap with dummy signatures so they show
                 *  up in the client.print_server_commands() function, but we need to intercept & process them as appropriate
                 */
                select cmd {
                    when "array"   { repTuple = arrayMsg(cmd, msgArgs, payload, st); }
                    when "connect" {
                        if authenticate {
                            repTuple = new MsgTuple("connected to arkouda server tcp://*:%i as user %s with token %s".format(
                                                            ServerPort,user,token), MsgType.NORMAL);
                        } else {
                            repTuple = new MsgTuple("connected to arkouda server tcp://*:%i".format(ServerPort), MsgType.NORMAL);
                        }
                    }
                    when "disconnect" {
                        if autoShutdown {
                            sendShutdownRequest(user=user);
                            break;
                        }
                        
                        repTuple = new MsgTuple("disconnected from arkouda server tcp://*:%i".format(ServerPort), MsgType.NORMAL);
                    }
                    when "noop" {
                        repTuple = new MsgTuple("noop", MsgType.NORMAL);
                    }
                    when "ruok" {
                        repTuple = new MsgTuple("imok", MsgType.NORMAL);
                    }
                    otherwise { // Look up in CommandMap or Binary CommandMap
                        if commandMap.contains(cmd) {
                            if moduleMap.contains(cmd) then
                                usedModules.add(moduleMap[cmd]);
                            repTuple = commandMap.getBorrowed(cmd)(cmd, msgArgs, st);
                        } else if commandMapBinary.contains(cmd) { // Binary response commands require different handling
                            if moduleMap.contains(cmd) then
                                usedModules.add(moduleMap[cmd]);
                            var binaryRepMsg = commandMapBinary.getBorrowed(cmd)(cmd, msgArgs, st);
                            sendRepMsg(binaryRepMsg);
                        } else {
                          repTuple = new MsgTuple("Unrecognized command: %s".format(cmd), MsgType.ERROR);
                          sdLogger.error(getModuleName(),getRoutineName(),getLineNumber(),repTuple.msg);
                        }
                    }
                }

                /*
                 * If the reply message is a string send it now
                 */          
                if !repTuple.msg.isEmpty() {
                    sendRepMsg(serialize(msg=repTuple.msg,msgType=repTuple.msgType,
                                                              msgFormat=MsgFormat.STRING, user=user));
                }

                /*
                 * log that the request message has been handled and reply message has been sent 
                 * along with the time to do so
                 */
                if trace {
                    sdLogger.info(getModuleName(),getRoutineName(),getLineNumber(), 
                                              "<<< %s took %.17r sec".format(cmd, t1.elapsed() - s0));
                }
                if (trace && memTrack) {
                    sdLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                        "bytes of memory used after command %t".format(getMemUsed():uint * numLocales:uint));
                }
                if metricsEnabled() {
                    userMetrics.incrementPerUserRequestMetrics(user,cmd);
                    requestMetrics.increment(cmd);
                }
            } catch (e: ErrorWithMsg) {
                // Generate a ReplyMsg of type ERROR and serialize to a JSON-formatted string
                sendRepMsg(serialize(msg=e.msg,msgType=MsgType.ERROR, msgFormat=MsgFormat.STRING, 
                                                        user=user));
                if trace {
                    sdLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                        "<<< %s resulted in error %s in  %.17r sec".format(cmd, e.msg, t1.elapsed() - s0));
                }
            } catch (e: Error) {
                // Generate a ReplyMsg of type ERROR and serialize to a JSON-formatted string
                var errorMsg = e.message();
            
                if errorMsg.isEmpty() {
                    errorMsg = "unexpected error";
                }

                sendRepMsg(serialize(msg=errorMsg,msgType=MsgType.ERROR, 
                                                         msgFormat=MsgFormat.STRING, user=user));
                if trace {
                    sdLogger.error(getModuleName(), getRoutineName(), getLineNumber(), 
                    "<<< %s resulted in error: %s in %.17r sec".format(cmd, e.message(),
                                                                                 t1.elapsed() - s0));
                }
            }
        }

        t1.stop();

        deleteServerConnectionInfo();

        sdLogger.info(getModuleName(), getRoutineName(), getLineNumber(),
            "requests = %i responseCount = %i elapsed sec = %i".format(reqCount,
                                                                       repCount,
                                                                       t1.elapsed()));
        this.shutdown(); 
        }
    }

    /**
     * The MetricsServerDaemon provides an endpoint for gathering user, request, 
     * locale, and server-scoped metrics
     */
    class MetricsServerDaemon : ArkoudaServerDaemon {
    
        var context: ZMQ.Context;
        var socket : ZMQ.Socket;      
        
        proc init() {
            this.socket = this.context.socket(ZMQ.REP); 
            this.port = try! getEnv('METRICS_SERVER_PORT','5556'):int;

            try! this.socket.bind("tcp://*:%t".format(this.port));
            try! sdLogger.debug(getModuleName(), 
                                getRoutineName(), 
                                getLineNumber(),
                                "initialized and listening in port %i".format(
                                this.port));
        }

        override proc run() throws {
            if integrationEnabled() {
                register(ServiceEndpoint.METRICS);
            }

            while !this.shutdownDaemon {
                sdLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                               "awaiting message on port %i".format(this.port));
                var req = this.socket.recv(bytes).decode();

                var msg: RequestMsg = extractRequest(req);
                var user   = msg.user;
                var token  = msg.token;
                var cmd    = msg.cmd;
                var format = msg.format;
                var args   = msg.args;
                var size   = msg.size: int;
                
                var msgArgs: owned MessageArgs;
                if size > 0 {
                    msgArgs = parseMessageArgs(args, size);
                }
                else {
                    msgArgs = new owned MessageArgs();
                }

                var repTuple: MsgTuple;

                select cmd {
                    when "metrics" {repTuple = metricsMsg(cmd, msgArgs, st);}        
                    when "connect" {
                        if authenticate {
                            repTuple = new MsgTuple("connected to arkouda metrics server tcp://*:%i as user " +
                                                "%s with token %s".format(this.port,user,token), MsgType.NORMAL);
                        } else {
                            repTuple = new MsgTuple("connected to arkouda metrics server tcp://*:%i".format(this.port), 
                                                                                    MsgType.NORMAL);
                        }
                    }
                    when "getconfig" {repTuple = getconfigMsg(cmd, msgArgs, st);}
                }           

                this.socket.send(serialize(msg=repTuple.msg,msgType=repTuple.msgType,
                                                msgFormat=MsgFormat.STRING, user=user));
            }

            if integrationEnabled() {
                deregisterFromExternalSystem(ServiceEndpoint.METRICS);
            }
            return;
        }
    }

    /**
     * The ExternalIntegrationServerDaemon class registers Arkouda with the
     * configured external system and then invokes ArkoudaServerDeamon.run()
     */    
    class ExternalIntegrationServerDaemon : DefaultServerDaemon {

        /**
         * Overridden run function registers the ServerDaemon with an 
         * external system and then invokes the parent run function.
         */
        override proc run() throws {
            register(ServiceEndpoint.ARKOUDA_CLIENT);
            super.run();
        }

        /**
         * Overridden shutdown function deregisters Arkouda from an external
         * system and then invokes the parent shutdown() function.
         */
        override proc shutdown() throws {
            on Locales[here.id] {
                deregisterFromExternalSystem(ServiceEndpoint.ARKOUDA_CLIENT);
                // if metrics is enabled, deregister the metrics socket
                if metricsEnabled() {
                    deregisterFromExternalSystem(ServiceEndpoint.METRICS);
                }
            }

            super.shutdown();
        }
    }

    proc getServerDaemon(daemonType: ServerDaemonType) : shared ArkoudaServerDaemon throws {
        select daemonType {
            when ServerDaemonType.DEFAULT {
                return new shared DefaultServerDaemon();
            }
            when ServerDaemonType.INTEGRATION {
                return new shared ExternalIntegrationServerDaemon();
            }
            when ServerDaemonType.METRICS {
               return new shared MetricsServerDaemon();
            }
            otherwise {
                throw getErrorWithContext(
                      msg="Unsupported ServerDaemonType: %t".format(daemonType),
                      lineNumber=getLineNumber(),
                      routineName=getRoutineName(),
                      moduleName=getModuleName(),
                      errorClass="IllegalArgumentError");
            }
        }
    }

    proc getServerDaemons() throws {
        var daemons = new list(shared ArkoudaServerDaemon);

        for (daemonType,i) in zip(serverDaemonTypes,0..serverDaemonTypes.size-1) {
            daemons.append(getServerDaemon(daemonType));
        }
        return daemons;
    }
}
