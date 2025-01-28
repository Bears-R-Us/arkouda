module ZarrMsg {
  use IO;
  use ServerErrors, ServerConfig;
  use FileIO;
  use FileSystem;
  use GenSymIO;
  use List;
  use Logging;
  use Message;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use NumPyDType;
  use Sort;
  use CommAggregation;
  use AryUtil;
  use CTypes;
  use Zarr;
  use ServerConfig;

  use Reflection;
  use ServerErrors;
  use ServerErrorStrings;
  use SegmentedString;

  use Map;
  use Math;

  @arkouda.instantiateAndRegister()
  proc readAllZarr(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    const storePath = msgArgs["store_path"].toScalar(string);

    const ar1 = readZarrArray(storePath, array_dtype, array_nd);
    var ar2 = makeDistArray((...ar1.shape), array_dtype);
    ar2 = ar1;

    return st.insert(new shared SymEntry(ar2));
  }

  @arkouda.instantiateAndRegister()
  proc writeAllZarr(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    const storePath = msgArgs.getValueOf("store_path"),
          chunkShape = msgArgs["chunk_shape"].toScalarTuple(int, array_nd);

    const ar1 = st[msgArgs["arr"]]: borrowed SymEntry(array_dtype, array_nd);
    writeZarrArray(storePath, ar1.a, chunkShape);

   return MsgTuple.success();
  }
}
