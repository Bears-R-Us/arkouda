module Message {
    use IO;
    use Reflection;
    use Errors;

    enum MsgType {NORMAL,WARNING,ERROR}
    enum MsgFormat {STRING,BINARY}

    /*
     * Encapsulates the message string and message type.
     */
    record MsgTuple {
        var msg: string;
        var msgType: MsgType;    
    }

    /*
     * Encapsulates state corresponding to a reply message sent back to 
     * the Arkouda client.
     */
    class ReplyMsg {
        var msg: string;
        var msgType: MsgType;
        var msgFormat: MsgFormat;
        var user: string;
    }

    /*
     * Encapsulates state corresponding to a client request sent to the Arkouda server.
     */
    record RequestMsg {
        var user: string;
        var token: string;
        var cmd: string;
        var format: string;
        var args: string;
    }

    /*
     * Deserializes a JSON-formatted string to a RequestMsg object, where the
     * JSON format is as follows:
     *
     * {"user": "user", "token": "token", "cmd": "cmd", "format": "STRING", "args": "arg1 arg2"}
     *
     */
    proc deserialize(ref msg: RequestMsg, request: string) throws {
        var newmem = openmem();
        newmem.writer().write(request);
        var nreader = newmem.reader();
        try {
            nreader.readf("%jt", msg);
        } catch bfe : BadFormatError {
            throw new owned ErrorWithContext("Incorrect JSON format %s".format(request),
                                       getLineNumber(),
                                       getRoutineName(),
                                       getModuleName(),
                                       "ValueError");
        }
    }
    
   /*
    * Generates a ReplyMsg object and serializes it into a JSON-formatted reply message
    */
   proc serialize(msg: string, msgType: MsgType, msgFormat: MsgFormat, 
                                                                 user: string) : string throws {
       return "%jt".format(new ReplyMsg(msg=msg,msgType=msgType, 
                                                        msgFormat=msgFormat, user=user));
   }
}