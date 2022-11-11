module Message {
    use IO;
    use FileIO;
    use Reflection;
    use ServerErrors;
    use NumPyDType;
    use Map;
    use List;

    enum MsgType {NORMAL,WARNING,ERROR}
    enum MsgFormat {STRING,BINARY}
    enum ObjectType {PDARRAY, SEGSTRING, LIST, DICT, VALUE, DATETIME, TIMEDELTA}

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
        var size: int; // currently unused, but wired for once all functionality moved to json
    }

    /*
    * Encapsulate parameter for a request sent to the Arkouda server
    * Note - Only used when args is in JSON format. 
    * Note - during the transition from space delimited string to JSON formated string, this object is not part of RequestMsg,
    *   but will be once all messages are transitioned to JSON arguments.
    */
    record ParameterObj {
        var key: string; // json key value 
        var val: string; // json value
        var objType: ObjectType; // type of the object
        var dtype: string; // type of elements contained in the object

        proc init() {}

        proc init(key: string, val: string, objType: ObjectType, dtype: string) {
            this.key = key;
            this.val = val;
            this.objType = objType;
            this.dtype = dtype;
        }

        proc asMap() throws {
            var m = new map(string, string);
            m.add("key", this.key);
            m.add("val", this.val);
            m.add("objType", this.objType:string);
            m.add("dtype", this.dtype);
            return m;
        }

        proc getJSON() throws {
            return "%jt".format(this);
        }

        proc setKey(value: string) {
            this.key = value;
        }

        proc setVal(value: string) {
            this.val = value;
        }

        proc setObjType(value: ObjectType) {
            this.ObjectType = value;
        }

        proc setDType(value: string) {
            this.dtype = value;
        }

        /*
        * Return the objType value
        * Returns str
        */
        proc getObjType(){
            return this.objType;
        }

        /*
        * Return the Dtype value as NumpyDtype
        * Returns Dtype
        */
        proc getDType() {
            return str2dtype(this.dtype);
        }

        /*
        * Return the raw string value
        * Returns string
        */
        proc getValue() {
            return this.val;
        }

        /*
        * Return the value as int64
        * Returns int
        */
        proc getIntValue(): int throws {
            try {
                return this.val:int;
            }
            catch {
                throw new owned ErrorWithContext("Parameter cannot be cast as int. Attempting to cast %s as type int failed".format(this.val),
                                    getLineNumber(),
                                    getRoutineName(),
                                    getModuleName(),
                                    "TypeError");
            }
        }

        /*
        * Return the value as uint64
        * Returns uint
        */
        proc getUIntValue(): uint throws {
            try {
                return this.val:uint;
            }
            catch {
                throw new owned ErrorWithContext("Parameter cannot be cast as uint. Attempting to cast %s as type uint failed".format(this.val),
                                    getLineNumber(),
                                    getRoutineName(),
                                    getModuleName(),
                                    "TypeError");
            }
        }

        proc getUInt8Value(): uint(8) throws {
            try {
                return this.val:uint(8);
            }
            catch {
                throw new owned ErrorWithContext("Parameter cannot be cast as uint(8). Attempting to cast %s as type uint(8) failed".format(this.val),
                                    getLineNumber(),
                                    getRoutineName(),
                                    getModuleName(),
                                    "TypeError");
            }
        }

        /*
        * Return the value as float64
        * Returns real
        */
        proc getRealValue(): real throws {
            try {
                return this.val:real;
            }
            catch {
                throw new owned ErrorWithContext("Parameter cannot be cast as real. Attempting to cast %s as type real failed".format(this.val),
                                    getLineNumber(),
                                    getRoutineName(),
                                    getModuleName(),
                                    "TypeError");
            }
        }

        /*
        * Return the value as bool
        * Returns bool
        */
        proc getBoolValue(): bool throws {
            try {
                return this.val.toLower():bool;
            }
            catch {
                throw new owned ErrorWithContext("Parameter cannot be cast as bool. Attempting to cast %s as type bool failed".format(this.val),
                                    getLineNumber(),
                                    getRoutineName(),
                                    getModuleName(),
                                    "TypeError");
            }
        }

        /*
        * Return the value as the provided type
        */
        proc getValueAsType(type t = string): t throws {
            if objType != ObjectType.VALUE {
                throw new owned ErrorWithContext("The value provided is not a castable type, please use ParameterObj.getSymEntry for this object.",
                                    getLineNumber(),
                                    getRoutineName(),
                                    getModuleName(),
                                    "TypeError");
            }
            
            try {
                return this.val:t; 
            }
            catch {
                throw new owned ErrorWithContext("Parameter cannot be cast as %t. Attempting to cast %s as type %t failed".format(t, this.val, t),
                                    getLineNumber(),
                                    getRoutineName(),
                                    getModuleName(),
                                    "TypeError");
            }
        }

        /*
        Parse value as a list of strings.
        :size: int: number of values in the list
        Note - not yet able to handle list of pdarray or SegString names
        */
        proc getList(size: int) throws {
            if this.objType != ObjectType.LIST {
                throw new owned ErrorWithContext("Parameter with key, %s, is not a list.".format(this.key),
                                    getLineNumber(),
                                    getRoutineName(),
                                    getModuleName(),
                                    "TypeError");
            }
            return jsonToPdArray(this.val, size);
        }

        proc getJSON(size: int) throws {
            if this.objType != ObjectType.DICT {
                throw new owned ErrorWithContext("Parameter with key, %s, is not a JSON obj.".format(this.key),
                                    getLineNumber(),
                                    getRoutineName(),
                                    getModuleName(),
                                    "TypeError");
            }
            return parseMessageArgs(this.val, size);
        }
    }

    /*
    Container class for the message arguments formatted as json
    :param_list:  array of ParameterObj
    :size: int - number of parameters contained in list
    */
    class MessageArgs {
        var param_list: list(ParameterObj);
        var size: int;

        proc init() {
            this.param_list = new list(ParameterObj);
            this.size = 0;
        }

        proc init(param_list: list(ParameterObj)) {
            this.param_list = param_list;
            this.size = param_list.size;
        }

        proc getJSON(keys: list(string) = list(string)): string throws {
            const noKeys: bool = keys.isEmpty();
            var s: int = if noKeys then this.size else keys.size;
            var json: [0..#s] string;
            var idx: int = 0;
            for p in this.param_list {
                if (noKeys || keys.contains(p.key)) {
                    json[idx] = p.getJSON();
                    idx += 1;
                }
                if idx > s {
                    break;
                }
            }
            return "%jt".format(json);
        }

        /*
        * Identify the parameter with the provided key and return it
        * Returns ParameterObj with the provided key
        * Throws KeyNotFound error if the provide key does not exist.
        */
        proc get(key: string) throws {
            for p in this.param_list {
                if p.key == key {
                    return p;
                }
            }
            throw new owned ErrorWithContext("Key Not Found; %s".format(key),
                                    getLineNumber(),
                                    getRoutineName(),
                                    getModuleName(),
                                    "KeyNotFound");
        }

        proc getValueOf(key: string) throws {
            for p in this.param_list {
                if p.key == key {
                    return p.val;
                }
            }
            throw new owned ErrorWithContext("Key Not Found; %s".format(key),
                                    getLineNumber(),
                                    getRoutineName(),
                                    getModuleName(),
                                    "KeyNotFound");
        }

        /*
        Return "iterable" of ParameterObj
        */
        proc items() {
            return this.param_list;
        }

        /*
        Return a list of all keys
        */
        proc keys() {
            var key_list: [0..#this.size] string;
            forall (idx, p) in zip(0..#this.size, this.param_list) {
                key_list[idx] = p.key;
            }
            return key_list;
        }

        /*
        Return a list of all values
        */
        proc vals(){
            var val_list: [0..#this.size] string;
            forall (idx, p) in zip(0..#this.size, this.param_list) {
                val_list[idx] = p.val;
            }
            return val_list;
        }

        /*
        Return bool if param_list contains given key name
        */
        proc contains(key: string): bool {
            var key_list = new list(this.keys());
            return key_list.contains(key);
        }
    }

    /*
    Parse and individual parameter components into a ParameterObj
    */
    proc parseParameter(payload:string) throws {
        var p: ParameterObj;
        var newmem = openmem();
        newmem.writer().write(payload);
        var nreader = newmem.reader();
        try {
            nreader.readf("%jt", p);
        } catch bfe : BadFormatError {
            throw new owned ErrorWithContext("Incorrect JSON format %s".format(payload),
                                       getLineNumber(),
                                       getRoutineName(),
                                       getModuleName(),
                                       "ValueError");
        }
        return p;
    }

    /*
    Parse arguments formatted as json string into objects
    */
    proc parseMessageArgs(json_str: string, size: int) throws {
        var pArr = jsonToPdArray(json_str, size);
        var param_list = new list(ParameterObj, parSafe=true);
        forall j_str in pArr with (ref param_list) {
            param_list.append(parseParameter(j_str));
        }
        return new owned MessageArgs(param_list);
    }

    /*
     * Deserializes a JSON-formatted string to a RequestMsg object, where the
     * JSON format is as follows (size is only set for json args. Otherwise, -1):
     *
     * {"user": "user", "token": "token", "cmd": "cmd", "format": "STRING", "args": "arg1 arg2", "size": "-1"}
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

    /*
     * Converts the JSON array to a pdarray
     */
    proc jsonToPdArray(json: string, size: int) throws {
        var f = opentmp(); defer { ensureClose(f); }
        var w = f.writer();
        w.write(json);
        w.close();
        var r = f.reader();
        var array: [0..#size] string;
        r.readf("%jt", array);
        r.close();
        return array;
    }

}
