module Message {
    use IO;
    use Reflection;
    use Errors;

    /*
     * Encapsulates state corresponding to a client request to Arkouda
     */
    record CommandMsg {
        var user: string;
        var token: string;
        var cmd: string;
        var format: string;
        var args: string;
    }

    /*
     * Deserializes a JSON-formatted string to a CommandMsg object, where the
     * JSON format is as follows:]
     *
     * {"user" : "user", "token" : "token", "cmd": "cmd", "format" : "STRING", "args": "arg1 arg2"}
     *
     */
    proc deserialize(ref msg: CommandMsg, cmd: string) throws {
        var newmem = openmem();
        newmem.writer().write(cmd);
        var nreader = newmem.reader();
        try {
            nreader.readf("%jt", msg);
        } catch bfe : BadFormatError {
            throw new owned ErrorWithContext("Incorrect JSON format %s".format(cmd),
                                       getLineNumber(),
                                       getRoutineName(),
                                       getModuleName(),
                                       "ValueError");
        }
    }
}