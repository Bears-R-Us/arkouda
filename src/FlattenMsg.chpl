module FlattenMsg {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Errors;
  use Reflection;
  use Flatten;
  use ServerConfig;
  use SegmentedArray;
  use GenSymIO;

  proc segFlattenMsg(cmd: string, payload: bytes, st: borrowed SymTab) throws {
    var (name, objtype, returnSegsStr, delimJson) = payload.decode().splitMsgToTuple(4);
    const returnSegs: bool = returnSegsStr.toLower() == "true";
    const arr = jsonToPdArray(delimJson, 1);
    const delim: string = arr[arr.domain.low];
    var repMsg: string;
    select objtype {
      when "str" {
        const rSegName = st.nextName();
        const rValName = st.nextName();
        const optName: string = if returnSegs then st.nextName() else "";
        var (segName, valName) = name.splitMsgToTuple('+', 2);
        const strings = new owned SegString(segName, valName, st);
        var (off, val, segs) = strings.flatten(delim, returnSegs);
        st.addEntry(rSegName, new shared SymEntry(off));
        st.addEntry(rValName, new shared SymEntry(val));
        repMsg = "created %s+created %s".format(st.attrib(rSegName), st.attrib(rValName));
        if returnSegs {
          st.addEntry(optName, new shared SymEntry(segs));
          repMsg += "+created %s".format(st.attrib(optName));
        }
      } otherwise {
        throw new owned ErrorWithContext("Not implemented for objtype %s".format(objtype),
                                         getLineNumber(),
                                         getRoutineName(),
                                         getModuleName(),
                                         "TypeError");
      }
    }
    return repMsg;
  }
}