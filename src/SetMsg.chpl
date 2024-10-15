module SetMsg {
  use Message;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use ServerConfig;
  use Logging;
  use ServerErrorStrings;
  use ServerErrors;
  use AryUtil;
  use CommAggregation;
  use RadixSortLSD;
  use Unique;
  use Reflection;
  use BigInteger;

  @arkouda.instantiateAndRegister
  proc uniqueValues(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws
    where (array_dtype != BigInteger.bigint) && (array_dtype != uint(8))
  {
    const name = msgArgs["name"],
          eIn = st[msgArgs["name"]]: SymEntry(array_dtype, array_nd),
          eFlat = if array_nd == 1 then eIn.a else flatten(eIn.a);

    const eSorted = radixSortLSD_keys(eFlat);
    const eUnique = uniqueFromSorted(eSorted, needCounts=false);

    return st.insert(new shared SymEntry(eUnique));
  }

  @arkouda.instantiateAndRegister
  proc uniqueCounts(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    const name = msgArgs.getValueOf("name"),
          eIn = st[msgArgs["name"]]: SymEntry(array_dtype, array_nd),
          eFlat = if array_nd == 1 then eIn.a else flatten(eIn.a);

    const eSorted = radixSortLSD_keys(eFlat);
    const (eUnique, eCounts) = uniqueFromSorted(eSorted);

    return MsgTuple.fromResponses([
                                    st.insert(new shared SymEntry(eUnique)),
                                    st.insert(new shared SymEntry(eCounts)),
                                  ]);
  }

  proc uniqueCounts(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws
    where (array_dtype == BigInteger.bigint) || (array_dtype == uint(8))
  {
      return MsgTuple.error("unique_counts does not support the %s dtype".format(array_dtype:string));
  }

  @arkouda.instantiateAndRegister
  proc uniqueInverse(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    const name = msgArgs.getValueOf("name"),
          eIn = st[msgArgs["name"]]: SymEntry(array_dtype, array_nd),
          eFlat = if array_nd == 1 then eIn.a else flatten(eIn.a);

    const (eUnique, _, inv) = uniqueSortWithInverse(eFlat);

    return MsgTuple.fromResponses([
                                    st.insert(new shared SymEntry(eUnique)),
                                    st.insert(new shared SymEntry(if array_nd == 1 then inv else unflatten(inv, eIn.a.shape))),
                                  ]);
  }

  proc uniqueInverse(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws
    where (array_dtype == BigInteger.bigint) || (array_dtype == uint(8))
  {
      return MsgTuple.error("unique_inverse does not support the %s dtype".format(array_dtype:string));
  }

  @arkouda.instantiateAndRegister
  proc uniqueAll(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    const name = msgArgs.getValueOf("name"),
          eIn = st[msgArgs["name"]]: SymEntry(array_dtype, array_nd),
          eFlat = if array_nd == 1 then eIn.a else flatten(eIn.a);

    const (eUnique, eCounts, inv, eIndices) = uniqueSortWithInverse(eFlat, needIndices=true);

    return MsgTuple.fromResponses([
                                    st.insert(new shared SymEntry(eUnique)),
                                    st.insert(new shared SymEntry(eIndices)),
                                    st.insert(new shared SymEntry(if array_nd == 1 then inv else unflatten(inv, eIn.a.shape))),
                                    st.insert(new shared SymEntry(eCounts)),
                                  ]);
  }

  proc uniqueAll(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws
    where (array_dtype == BigInteger.bigint) || (array_dtype == uint(8))
  {
      return MsgTuple.error("unique_all does not support the %s dtype".format(array_dtype:string));
  }
}
