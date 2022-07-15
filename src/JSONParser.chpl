module JSONParser {
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerConfig;
    use ServerErrors;
    use Reflection;
    use SegmentedString;

    proc readJSONMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        var j: string;
        writeln("\nVALUE OF JSONARRAY: %s\n".format(payload));
        var msgArgs = parseMessageArgs(payload, 2);
        
        writeln("\n%jt\n".format(msgArgs));

        for p in msgArgs.items() {
            writeln("Key: %t, Val: %t, ObjType: %s, DType: %s\n".format(p.key, p.getValue(), p.getObjType(), p.getDType()));
        }

        writeln("\nKeys: %jt\n".format(msgArgs.keys()));
        writeln("\nVals: %jt\n".format(msgArgs.vals()));

        var a = msgArgs.get("akarray");
        
        var genSym: borrowed GenSymEntry = getGenericTypedArrayEntry(a.val, st);
        var p = toSymEntry(genSym, int);
        writeln("\nARRAY");
        for x in p.a {
            writeln("%s".format(x));
        }

        var s = msgArgs.get("segstr");
        var ss = getSegString(s.val, st);
        writeln("\nSTRINGS");
        for x in ss.offsets.a {
            writeln("%s".format(x));
        }

        return new MsgTuple("Test Complete", MsgType.NORMAL);     
    }

    proc registerMe() {
        use CommandMap;
        registerFunction("readJSON", readJSONMsg, getModuleName());
    }
}