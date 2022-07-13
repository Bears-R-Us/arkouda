module ServerDaemon {
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
    use CommandMap, ServerRegistration;

    private config const logLevel = ServerConfig.logLevel;
    const sdLogger = new Logger(logLevel);

    class ArkoudaServerDaemon {
        var st = new owned SymTab();
        var shutdownServer = false;
        var serverToken : string;
        var serverMessage : string;
        var reqCount: int = 0;
        var repCount: int = 0;
        
        var context: ZMQ.Context;
        var socket : ZMQ.Socket;        
       
        
        proc init() {
            this.socket = this.context.socket(ZMQ.REP); 
            try! this.socket.bind("tcp://*:%t".format(ServerPort));
        }
        
        proc run() {
        
        }
    }
}
