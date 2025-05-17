module Message {
    use IO;
    use JSON;
    use IOUtils;
    use FileIO;
    use Reflection;
    use ServerErrors;
    use NumPyDType;
    use List;
    use BigInteger;
    use MultiTypeSymEntry;
    use Map;

    enum MsgType {NORMAL,WARNING,ERROR}
    enum MsgFormat {STRING,BINARY}

    /*
     * Encapsulates the message string and message type.
     */
    record MsgTuple {
        var msg: string;
        var msgType: MsgType;
        var msgFormat: MsgFormat;
        var user: string;
        var payload: bytes;
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

    proc MsgTuple.init() {
        this.msg = "";
        this.msgType = MsgType.NORMAL;
        this.msgFormat = MsgFormat.STRING;
        this.user = "";
        this.payload = b"";
    }

    // deprecated in favor of type-method factories below
    proc MsgTuple.init(msg: string, msgType: MsgType) {
        this.msg = msg;
        this.msgType = msgType;
        this.msgFormat = MsgFormat.STRING;
        this.user = "";
        this.payload = b"";
    }

    proc MsgTuple.init(msg: string, msgType: MsgType, msgFormat: MsgFormat, user = "", in payload = b"") {
        this.msg = msg;
        this.msgType = msgType;
        this.msgFormat = msgFormat;
        this.user = "";
        this.payload = payload;
    }

    proc type MsgTuple.success(msg: string = ""): MsgTuple {
        return new MsgTuple(
            msg = msg,
            msgType = MsgType.NORMAL,
            msgFormat = MsgFormat.STRING,
            payload = b""
        );
    }

    proc type MsgTuple.warning(msg: string): MsgTuple {
        return new MsgTuple(
            msg = msg,
            msgType = MsgType.WARNING,
            msgFormat = MsgFormat.STRING,
            payload = b""
        );
    }

    proc type MsgTuple.error(msg: string): MsgTuple {
        return new MsgTuple(
            msg = msg,
            msgType = MsgType.ERROR,
            msgFormat = MsgFormat.STRING,
            payload = b""
        );
    }

    /*
        Create a MsgTuple indicating to the client that a new symbol was created
    */
    proc type MsgTuple.newSymbol(name: string, sym: borrowed AbstractSymEntry): MsgTuple throws {
        var msg = "created " + name + " ";

        if sym.isAssignableTo(SymbolEntryType.TypedArraySymEntry) {
            msg += (sym: borrowed GenSymEntry).attrib();
        } else if sym.isAssignableTo(SymbolEntryType.CompositeSymEntry) {
            msg += (sym: borrowed CompositeSymEntry).attrib();
        } else if sym.isAssignableTo(SymbolEntryType.SparseSymEntry) {
            msg += (sym: borrowed GenSparseSymEntry).attrib();
        }

        return new MsgTuple(
            msg = msg,
            msgType = MsgType.NORMAL,
            msgFormat = MsgFormat.STRING,
            payload = b""
        );
    }

    /*
        Create a MsgTuple from a group of responses (useful for returning multiple
        symbols from one command, see: 'unstack' in 'ManipulationMsg')

        If any of the responses are errors, return the first error message.
        Otherwise, return a success message, where each of the 'msg' fields
        are composed into a JSON list.
    */
    proc type MsgTuple.fromResponses(responses: [] MsgTuple): MsgTuple throws {
        for res in responses {
            if res.msgType == MsgType.ERROR {
                return res;
            }
        }

        var msgs = new list([res in responses] res.msg);

        return new MsgTuple(
            msg = formatJson(msgs),
            msgType = MsgType.NORMAL,
            msgFormat = MsgFormat.STRING,
            payload = b""
        );
    }

    proc type MsgTuple.fromScalar(scalar: ?t): MsgTuple throws {
        import NumPyDType;
        const dTypeName = type2str(t);
        if dTypeName == "undef"
            then throw new Error("Unknown scalar type '%s' in MsgTuple.fromScalar".format(t:string));
        return new MsgTuple(
            msg = "%s %s".format(dTypeName, NumPyDType.type2fmt(t)).format(scalar),
            msgType = MsgType.NORMAL,
            msgFormat = MsgFormat.STRING,
            payload = b""
        );
    }

    proc type MsgTuple.payload(in data: bytes): MsgTuple {
        return new MsgTuple(
            msg = "",
            msgType = MsgType.NORMAL,
            msgFormat = MsgFormat.BINARY,
            payload = data
        );
    }

    proc ref MsgTuple.serialize(user: string) throws {
        this.user = user;
        return formatJson(this);
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
        var dtype: string; // type of elements contained in the object

        proc init() {}

        proc init(key: string, val: string, dtype: string) {
            this.key = key;
            this.val = val;
            this.dtype = dtype;
        }

        proc ref setKey(value: string) {
            this.key = value;
        }

        proc ref setVal(value: string) {
            this.val = value;
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

        // temporary implementation until more robust JSON parsing is implemented
        proc tryGetScalar(type t): (t, bool) {
            if t == string {
                // is this a number? if so, assume it isn't a string
                try {
                    const x = this.val: real;
                    return ("", false);
                } catch {
                    return (this.val, true);
                }
            } else {
                try {
                    return (this.val: t, true);
                } catch {
                    return (t.init, false);
                }
            }
        }

        /*
            Get a value of the given type from a JSON scalar

            :arg t: type: the scalar type to parse

            Throws an ErrorWithContext if the value cannot be parsed as the given
                type
        */
        proc toScalar(type t): t throws {
            try {
                if t == string {
                    return this.val;
                } else {
                    // temporary special case (until frontend is modified to provide lowercase 'true'/'false' values)
                    if t == bool {
                        if this.val.startsWith("True") then return true;
                        if this.val.startsWith("False") then return false;
                        return this.val:int:bool;
                    }
                    return this.val: t;
                }
            } catch {
                throw new ErrorWithContext(
                    "Json Argument '%s' cannot be cast as a scalar %s".format(this.val, t:string),
                    getLineNumber(),
                    getRoutineName(),
                    getModuleName(),
                    "TypeError"
                );
            }
        }

        /*
            Get a tuple of the given type from a JSON array of scalar values

            :arg size: int: the number of elements in the tuple (must match
                the number of elements in the JSON array)
            :arg t: type: the scalar type to parse

            Throws an ErrorWithContext if the value is not a JSON array, the
                tuple is the wrong size, or if the elements of the array cannot
                be parsed as the given type
        */
        proc toScalarTuple(type t, param size: int): size*t throws {
            proc err() throws {
                return new ErrorWithContext(
                    "Json Argument '%s' cannot be cast as a %i-tuple of %s".format(this.val, size, t:string),
                    getLineNumber(),
                    getRoutineName(),
                    getModuleName(),
                    "TypeError"
                );
            }

            if size == 1 {
                // special case to support parsing a scalar as a 1-tuple
                try {
                    return (this.toScalar(t),);
                } catch {
                    try {
                        return parseJson(this.val, 1, t);
                    } catch {
                        throw err();
                    }
                }
            } else {
                try {
                    return parseJson(this.val, size, t);
                } catch {
                    throw err();
                }
            }
        }

        /*
            Get a list of the given type from a JSON array of scalar values

            :arg t: type: the scalar type to parse

            Throws an ErrorWithContext if the value is not a JSON array or if the
                elements of the array cannot be parsed as the given type
        */
        proc toScalarList(type t): list(t) throws {
            proc err() throws {
                return new ErrorWithContext(
                    "Json Argument '%s' cannot be cast as a list of %s".format(this.val, t:string),
                    getLineNumber(),
                    getRoutineName(),
                    getModuleName(),
                    "TypeError"
                );
            }

            try {
                const sl = parseJson(this.val, list(string));
                var l = new list(t);
                for s in sl do l.pushBack(s: t);
                return l;
            } catch {
                throw err();
            }
        }

        /*
            Get an array of the given type from a JSON array of scalar values

            :arg t: type: the scalar type to parse
            :arg size: int: the number of elements in the array (must match the
                number of elements in the JSON array)

            Throws an ErrorWithContext if the value is not a JSON array, the
                array is the wrong size, or if the elements of the array cannot
                be parsed as the given type
        */
        proc toScalarArray(type t, size: int): [] t throws {
            proc err() throws {
                return new ErrorWithContext(
                    "Json Argument '%s' cannot be cast as an array of %s".format(this.val, t:string),
                    getLineNumber(),
                    getRoutineName(),
                    getModuleName(),
                    "TypeError"
                );
            }

            try {
                return jsonToArray(this.val, t, size);
            } catch {
                throw err();
            }
        }

        /*
         * Attempt to cast the value to the provided type
         * Throw and error if the cast isn't possible
        */
        // deprecated
        proc getScalarValue(type t): t throws {
            return this.toScalar(t);
        }

        /*
        * Return the value as int64
        * Returns int
        */
        // deprecated
        proc getIntValue(): int throws {
            return this.toScalar(int);
        }

        /*
            Return the value as a positive int
            If the value is negative, return 'value + max + 1', otherwise return the value
            This is useful for implementing Python's negative indexing rules
        */
        proc getPositiveIntValue(max: int): int throws {
            var x = this.toScalar(int);
            if x >= 0 && x < max then return x;
            if x < 0 && x >= -max then return x + max;
            else throw new ErrorWithContext(
                "Parameter cannot be cast as a positive int in the range: [%?, %?)".format(-max, max),
                getLineNumber(),
                getRoutineName(),
                getModuleName(),
                "ValueError"
            );
        }

        /*
        * Return the value as uint64
        * Returns uint
        */
        // deprecated
        proc getUIntValue(): uint throws {
            return this.toScalar(uint);
        }

        // deprecated
        proc getUInt8Value(): uint(8) throws {
            return this.toScalar(uint(8));
        }

        /*
        * Return the value as float64
        * Returns real
        */
        // deprecated
        proc getRealValue(): real throws {
            return this.toScalar(real);
        }

        /*
        * Return the value as bool
        * Returns bool
        */
        // deprecated
        proc getBoolValue(): bool throws {
            return this.toScalar(bool);
        }

        // deprecated
        proc getBigIntValue(): bigint throws {
            return this.toScalar(bigint);
        }

        /*
        Parse value as a list of strings.
        :size: int: number of values in the list
        Note - not yet able to handle list of pdarray or SegString names
        */
        // deprecated
        proc getList(size: int) throws {
            return this.toScalarArray(string, size);
        }

        /*
            Parse value as a tuple of integers with the given size
        */
        // deprecated
        proc getTuple(param size: int): size*int throws {
            return this.toScalarTuple(int, size);
        }
    }

    /*
    Container class for the message arguments formatted as json
    :param_list:  array of ParameterObj
    :size: int - number of parameters contained in list
    */
    class MessageArgs: writeSerializable {
        // TODO: reimplement argument representation using a JSON class hierarchy (store the head of the JSON tree here)
        var param_list: list(ParameterObj);
        var size: int;
        var payload: bytes;

        proc init() {
            this.param_list = new list(ParameterObj);
            this.size = 0;
            this.payload = b"";
        }

        proc init(param_list: list(ParameterObj)) {
            this.param_list = param_list;
            this.size = param_list.size;
            this.payload = b"";
        }

        proc init(param_list: list(ParameterObj, parSafe=true), in payload: bytes) {
            // Intentionally initializes the param_list with `parSafe=false`.
            // It would be initialized that way anyways due to the field
            // declaration relying on the default value, this just makes it
            // explicit (and avoids a warning as a result).
            this.param_list = new list(ParameterObj);
            this.size = param_list.size;

            this.param_list = param_list;
            this.payload = payload;
        }

        /*
            Attach a binary payload to the message arguments
        */
        proc addPayload(in p: bytes) {
            this.payload = p;
        }

        /*
            Get the value of the argument with the given key
        */
        proc this(key: string): ParameterObj throws {
            for p in this.param_list {
                if p.key == key {
                    return p;
                }
            }

            throw new owned ErrorWithContext("JSON argument key Not Found: %s".format(key),
                                             getLineNumber(),
                                             getRoutineName(),
                                             getModuleName(),
                                             "KeyNotFound");
        }

        iter these(): ParameterObj {
            for p in this.param_list {
                yield p;
            }
        }

        override proc serialize(writer: fileWriter(?), ref serializer: ?st) throws {
            var ser = serializer.startClass(writer, "MessageArgs", 3);
            ser.writeField("param_list", this.param_list);
            ser.writeField("size", this.size);
            ser.writeField("payload", if this.payload.size > 0 then "<binary_payload>" else "");
            ser.endClass();
        }

        /*
        * Identify the parameter with the provided key and return it
        * Returns ParameterObj with the provided key
        * Throws KeyNotFound error if the provide key does not exist.
        */
        proc get(key: string): ParameterObj throws {
            return this[key];
        }

        proc getValueOf(key: string): string throws {
            return this[key].val;
        }

        /*
          Return true if there is an argument with the given name, false otherwise
        */
        proc contains(key: string): bool {
            for p in this.param_list {
                if p.key == key {
                    return true;
                }
            }
            return false;
        }
    }

    /*
    Parse and individual parameter components into a ParameterObj
    */
    proc parseParameter(payload:string) throws {
        var p: ParameterObj;
        var newmem = openMemFile();
        newmem.writer(locking=false).write(payload);
        try {
            var nreader = newmem.reader(deserializer=new jsonDeserializer(), locking=false);
            nreader.readf("%?", p);
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
    proc parseMessageArgs(json_str: string, size: int, in payload = b"") throws {
        var pArr = jsonToArray(json_str, string, size);
        var param_list = new list(ParameterObj, parSafe=true);
        forall j_str in pArr with (ref param_list) {
            param_list.pushBack(parseParameter(j_str));
        }
        return new owned MessageArgs(param_list, payload);
    }

    /*
     * Deserializes a JSON-formatted string to a RequestMsg object, where the
     * JSON format is as follows (size is only set for json args. Otherwise, -1):
     *
     * {"user": "user", "token": "token", "cmd": "cmd", "format": "STRING", "args": "arg1 arg2", "size": "-1"}
     *
     */
    proc deserialize(ref msg: RequestMsg, request: string) throws {
        var newmem = openMemFile();
        newmem.writer(locking=false).write(request);
        try {
            var nreader = newmem.reader(deserializer=new jsonDeserializer(), locking=false);
            nreader.readf("%?", msg);
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
       return formatJson(new MsgTuple(msg=msg,msgType=msgType,
                                      msgFormat=msgFormat, user=user));
   }
}
