module Commands {

use CommandMap, Message, MultiTypeSymbolTable, MultiTypeSymEntry;

use BigInteger;

param regConfig = """
{
  "parameter_classes": {
    "array": {
      "nd": [
        1
      ],
      "dtype": [
        "int",
        "uint",
        "uint(8)",
        "real",
        "bool",
        "bigint"
      ]
    }
  }
}
""";

import CastMsg;

proc ark_cast_int_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=int, array_nd=1);
registerFunction('cast<int64,int64,1>', ark_cast_int_int_1, 'CastMsg', 23);

proc ark_cast_int_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=uint, array_nd=1);
registerFunction('cast<int64,uint64,1>', ark_cast_int_uint_1, 'CastMsg', 23);

proc ark_cast_int_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=uint(8), array_nd=1);
registerFunction('cast<int64,uint8,1>', ark_cast_int_uint8_1, 'CastMsg', 23);

proc ark_cast_int_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=real, array_nd=1);
registerFunction('cast<int64,float64,1>', ark_cast_int_real_1, 'CastMsg', 23);

proc ark_cast_int_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=bool, array_nd=1);
registerFunction('cast<int64,bool,1>', ark_cast_int_bool_1, 'CastMsg', 23);

proc ark_cast_int_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=bigint, array_nd=1);
registerFunction('cast<int64,bigint,1>', ark_cast_int_bigint_1, 'CastMsg', 23);

proc ark_cast_uint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=int, array_nd=1);
registerFunction('cast<uint64,int64,1>', ark_cast_uint_int_1, 'CastMsg', 23);

proc ark_cast_uint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=uint, array_nd=1);
registerFunction('cast<uint64,uint64,1>', ark_cast_uint_uint_1, 'CastMsg', 23);

proc ark_cast_uint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=uint(8), array_nd=1);
registerFunction('cast<uint64,uint8,1>', ark_cast_uint_uint8_1, 'CastMsg', 23);

proc ark_cast_uint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=real, array_nd=1);
registerFunction('cast<uint64,float64,1>', ark_cast_uint_real_1, 'CastMsg', 23);

proc ark_cast_uint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=bool, array_nd=1);
registerFunction('cast<uint64,bool,1>', ark_cast_uint_bool_1, 'CastMsg', 23);

proc ark_cast_uint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=bigint, array_nd=1);
registerFunction('cast<uint64,bigint,1>', ark_cast_uint_bigint_1, 'CastMsg', 23);

proc ark_cast_uint8_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=int, array_nd=1);
registerFunction('cast<uint8,int64,1>', ark_cast_uint8_int_1, 'CastMsg', 23);

proc ark_cast_uint8_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=uint, array_nd=1);
registerFunction('cast<uint8,uint64,1>', ark_cast_uint8_uint_1, 'CastMsg', 23);

proc ark_cast_uint8_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=uint(8), array_nd=1);
registerFunction('cast<uint8,uint8,1>', ark_cast_uint8_uint8_1, 'CastMsg', 23);

proc ark_cast_uint8_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=real, array_nd=1);
registerFunction('cast<uint8,float64,1>', ark_cast_uint8_real_1, 'CastMsg', 23);

proc ark_cast_uint8_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=bool, array_nd=1);
registerFunction('cast<uint8,bool,1>', ark_cast_uint8_bool_1, 'CastMsg', 23);

proc ark_cast_uint8_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=bigint, array_nd=1);
registerFunction('cast<uint8,bigint,1>', ark_cast_uint8_bigint_1, 'CastMsg', 23);

proc ark_cast_real_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=int, array_nd=1);
registerFunction('cast<float64,int64,1>', ark_cast_real_int_1, 'CastMsg', 23);

proc ark_cast_real_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=uint, array_nd=1);
registerFunction('cast<float64,uint64,1>', ark_cast_real_uint_1, 'CastMsg', 23);

proc ark_cast_real_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=uint(8), array_nd=1);
registerFunction('cast<float64,uint8,1>', ark_cast_real_uint8_1, 'CastMsg', 23);

proc ark_cast_real_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=real, array_nd=1);
registerFunction('cast<float64,float64,1>', ark_cast_real_real_1, 'CastMsg', 23);

proc ark_cast_real_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=bool, array_nd=1);
registerFunction('cast<float64,bool,1>', ark_cast_real_bool_1, 'CastMsg', 23);

proc ark_cast_real_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=bigint, array_nd=1);
registerFunction('cast<float64,bigint,1>', ark_cast_real_bigint_1, 'CastMsg', 23);

proc ark_cast_bool_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=int, array_nd=1);
registerFunction('cast<bool,int64,1>', ark_cast_bool_int_1, 'CastMsg', 23);

proc ark_cast_bool_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=uint, array_nd=1);
registerFunction('cast<bool,uint64,1>', ark_cast_bool_uint_1, 'CastMsg', 23);

proc ark_cast_bool_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=uint(8), array_nd=1);
registerFunction('cast<bool,uint8,1>', ark_cast_bool_uint8_1, 'CastMsg', 23);

proc ark_cast_bool_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=real, array_nd=1);
registerFunction('cast<bool,float64,1>', ark_cast_bool_real_1, 'CastMsg', 23);

proc ark_cast_bool_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=bool, array_nd=1);
registerFunction('cast<bool,bool,1>', ark_cast_bool_bool_1, 'CastMsg', 23);

proc ark_cast_bool_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=bigint, array_nd=1);
registerFunction('cast<bool,bigint,1>', ark_cast_bool_bigint_1, 'CastMsg', 23);

proc ark_cast_bigint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=int, array_nd=1);
registerFunction('cast<bigint,int64,1>', ark_cast_bigint_int_1, 'CastMsg', 23);

proc ark_cast_bigint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=uint, array_nd=1);
registerFunction('cast<bigint,uint64,1>', ark_cast_bigint_uint_1, 'CastMsg', 23);

proc ark_cast_bigint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=uint(8), array_nd=1);
registerFunction('cast<bigint,uint8,1>', ark_cast_bigint_uint8_1, 'CastMsg', 23);

proc ark_cast_bigint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=real, array_nd=1);
registerFunction('cast<bigint,float64,1>', ark_cast_bigint_real_1, 'CastMsg', 23);

proc ark_cast_bigint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=bool, array_nd=1);
registerFunction('cast<bigint,bool,1>', ark_cast_bigint_bool_1, 'CastMsg', 23);

proc ark_cast_bigint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=bigint, array_nd=1);
registerFunction('cast<bigint,bigint,1>', ark_cast_bigint_bigint_1, 'CastMsg', 23);

proc ark_castToStrings_int(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArrayToStrings(cmd, msgArgs, st, array_dtype=int);
registerFunction('castToStrings<int64>', ark_castToStrings_int, 'CastMsg', 60);

proc ark_castToStrings_uint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArrayToStrings(cmd, msgArgs, st, array_dtype=uint);
registerFunction('castToStrings<uint64>', ark_castToStrings_uint, 'CastMsg', 60);

proc ark_castToStrings_uint8(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArrayToStrings(cmd, msgArgs, st, array_dtype=uint(8));
registerFunction('castToStrings<uint8>', ark_castToStrings_uint8, 'CastMsg', 60);

proc ark_castToStrings_real(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArrayToStrings(cmd, msgArgs, st, array_dtype=real);
registerFunction('castToStrings<float64>', ark_castToStrings_real, 'CastMsg', 60);

proc ark_castToStrings_bool(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArrayToStrings(cmd, msgArgs, st, array_dtype=bool);
registerFunction('castToStrings<bool>', ark_castToStrings_bool, 'CastMsg', 60);

proc ark_castToStrings_bigint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArrayToStrings(cmd, msgArgs, st, array_dtype=bigint);
registerFunction('castToStrings<bigint>', ark_castToStrings_bigint, 'CastMsg', 60);

proc ark_castStringsTo_int(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castStringsToArray(cmd, msgArgs, st, array_dtype=int);
registerFunction('castStringsTo<int64>', ark_castStringsTo_int, 'CastMsg', 67);

proc ark_castStringsTo_uint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castStringsToArray(cmd, msgArgs, st, array_dtype=uint);
registerFunction('castStringsTo<uint64>', ark_castStringsTo_uint, 'CastMsg', 67);

proc ark_castStringsTo_uint8(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castStringsToArray(cmd, msgArgs, st, array_dtype=uint(8));
registerFunction('castStringsTo<uint8>', ark_castStringsTo_uint8, 'CastMsg', 67);

proc ark_castStringsTo_real(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castStringsToArray(cmd, msgArgs, st, array_dtype=real);
registerFunction('castStringsTo<float64>', ark_castStringsTo_real, 'CastMsg', 67);

proc ark_castStringsTo_bool(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castStringsToArray(cmd, msgArgs, st, array_dtype=bool);
registerFunction('castStringsTo<bool>', ark_castStringsTo_bool, 'CastMsg', 67);

proc ark_castStringsTo_bigint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castStringsToArray(cmd, msgArgs, st, array_dtype=bigint);
registerFunction('castStringsTo<bigint>', ark_castStringsTo_bigint, 'CastMsg', 67);

import IndexingMsg;

proc ark_reg_intIndex_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int): MsgTuple throws {
  var a_array_sym = st[msgArgs['a']]: SymEntry(array_dtype_0, array_nd_0);
  ref a = a_array_sym.a;
  var idx = msgArgs['idx'].toScalarTuple(int, array_nd_0);
  var ark_result = IndexingMsg.intIndex(a,idx);

  return MsgTuple.fromScalar(ark_result);
}

proc ark__int__int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('[int]<int64,1>', ark__int__int_1, 'IndexingMsg', 197);

proc ark__int__uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('[int]<uint64,1>', ark__int__uint_1, 'IndexingMsg', 197);

proc ark__int__uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('[int]<uint8,1>', ark__int__uint8_1, 'IndexingMsg', 197);

proc ark__int__real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('[int]<float64,1>', ark__int__real_1, 'IndexingMsg', 197);

proc ark__int__bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('[int]<bool,1>', ark__int__bool_1, 'IndexingMsg', 197);

proc ark__int__bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('[int]<bigint,1>', ark__int__bigint_1, 'IndexingMsg', 197);

proc ark_arrayViewIntIndex_int(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.arrayViewIntIndex(cmd, msgArgs, st, array_dtype=int);
registerFunction('arrayViewIntIndex<int64>', ark_arrayViewIntIndex_int, 'IndexingMsg', 117);

proc ark_arrayViewIntIndex_uint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.arrayViewIntIndex(cmd, msgArgs, st, array_dtype=uint);
registerFunction('arrayViewIntIndex<uint64>', ark_arrayViewIntIndex_uint, 'IndexingMsg', 117);

proc ark_arrayViewIntIndex_uint8(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.arrayViewIntIndex(cmd, msgArgs, st, array_dtype=uint(8));
registerFunction('arrayViewIntIndex<uint8>', ark_arrayViewIntIndex_uint8, 'IndexingMsg', 117);

proc ark_arrayViewIntIndex_real(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.arrayViewIntIndex(cmd, msgArgs, st, array_dtype=real);
registerFunction('arrayViewIntIndex<float64>', ark_arrayViewIntIndex_real, 'IndexingMsg', 117);

proc ark_arrayViewIntIndex_bool(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.arrayViewIntIndex(cmd, msgArgs, st, array_dtype=bool);
registerFunction('arrayViewIntIndex<bool>', ark_arrayViewIntIndex_bool, 'IndexingMsg', 117);

proc ark_arrayViewIntIndex_bigint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.arrayViewIntIndex(cmd, msgArgs, st, array_dtype=bigint);
registerFunction('arrayViewIntIndex<bigint>', ark_arrayViewIntIndex_bigint, 'IndexingMsg', 117);

import ManipulationMsg;

proc ark_broadcast_int_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<int64,1,1>', ark_broadcast_int_1_1, 'ManipulationMsg', 61);

proc ark_broadcast_uint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<uint64,1,1>', ark_broadcast_uint_1_1, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<uint8,1,1>', ark_broadcast_uint8_1_1, 'ManipulationMsg', 61);

proc ark_broadcast_real_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<float64,1,1>', ark_broadcast_real_1_1, 'ManipulationMsg', 61);

proc ark_broadcast_bool_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<bool,1,1>', ark_broadcast_bool_1_1, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<bigint,1,1>', ark_broadcast_bigint_1_1, 'ManipulationMsg', 61);

proc ark_concat_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('concat<int64,1>', ark_concat_int_1, 'ManipulationMsg', 158);

proc ark_concat_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('concat<uint64,1>', ark_concat_uint_1, 'ManipulationMsg', 158);

proc ark_concat_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('concat<uint8,1>', ark_concat_uint8_1, 'ManipulationMsg', 158);

proc ark_concat_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('concat<float64,1>', ark_concat_real_1, 'ManipulationMsg', 158);

proc ark_concat_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('concat<bool,1>', ark_concat_bool_1, 'ManipulationMsg', 158);

proc ark_concat_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('concat<bigint,1>', ark_concat_bigint_1, 'ManipulationMsg', 158);

proc ark_concatFlat_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('concatFlat<int64,1>', ark_concatFlat_int_1, 'ManipulationMsg', 214);

proc ark_concatFlat_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('concatFlat<uint64,1>', ark_concatFlat_uint_1, 'ManipulationMsg', 214);

proc ark_concatFlat_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('concatFlat<uint8,1>', ark_concatFlat_uint8_1, 'ManipulationMsg', 214);

proc ark_concatFlat_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('concatFlat<float64,1>', ark_concatFlat_real_1, 'ManipulationMsg', 214);

proc ark_concatFlat_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('concatFlat<bool,1>', ark_concatFlat_bool_1, 'ManipulationMsg', 214);

proc ark_concatFlat_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('concatFlat<bigint,1>', ark_concatFlat_bigint_1, 'ManipulationMsg', 214);

proc ark_expandDims_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('expandDims<int64,1>', ark_expandDims_int_1, 'ManipulationMsg', 238);

proc ark_expandDims_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('expandDims<uint64,1>', ark_expandDims_uint_1, 'ManipulationMsg', 238);

proc ark_expandDims_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('expandDims<uint8,1>', ark_expandDims_uint8_1, 'ManipulationMsg', 238);

proc ark_expandDims_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('expandDims<float64,1>', ark_expandDims_real_1, 'ManipulationMsg', 238);

proc ark_expandDims_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('expandDims<bool,1>', ark_expandDims_bool_1, 'ManipulationMsg', 238);

proc ark_expandDims_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('expandDims<bigint,1>', ark_expandDims_bigint_1, 'ManipulationMsg', 238);

proc ark_flip_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('flip<int64,1>', ark_flip_int_1, 'ManipulationMsg', 290);

proc ark_flip_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('flip<uint64,1>', ark_flip_uint_1, 'ManipulationMsg', 290);

proc ark_flip_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('flip<uint8,1>', ark_flip_uint8_1, 'ManipulationMsg', 290);

proc ark_flip_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('flip<float64,1>', ark_flip_real_1, 'ManipulationMsg', 290);

proc ark_flip_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('flip<bool,1>', ark_flip_bool_1, 'ManipulationMsg', 290);

proc ark_flip_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('flip<bigint,1>', ark_flip_bigint_1, 'ManipulationMsg', 290);

proc ark_flipAll_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('flipAll<int64,1>', ark_flipAll_int_1, 'ManipulationMsg', 358);

proc ark_flipAll_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('flipAll<uint64,1>', ark_flipAll_uint_1, 'ManipulationMsg', 358);

proc ark_flipAll_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('flipAll<uint8,1>', ark_flipAll_uint8_1, 'ManipulationMsg', 358);

proc ark_flipAll_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('flipAll<float64,1>', ark_flipAll_real_1, 'ManipulationMsg', 358);

proc ark_flipAll_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('flipAll<bool,1>', ark_flipAll_bool_1, 'ManipulationMsg', 358);

proc ark_flipAll_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('flipAll<bigint,1>', ark_flipAll_bigint_1, 'ManipulationMsg', 358);

proc ark_permuteDims_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('permuteDims<int64,1>', ark_permuteDims_int_1, 'ManipulationMsg', 389);

proc ark_permuteDims_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('permuteDims<uint64,1>', ark_permuteDims_uint_1, 'ManipulationMsg', 389);

proc ark_permuteDims_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('permuteDims<uint8,1>', ark_permuteDims_uint8_1, 'ManipulationMsg', 389);

proc ark_permuteDims_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('permuteDims<float64,1>', ark_permuteDims_real_1, 'ManipulationMsg', 389);

proc ark_permuteDims_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('permuteDims<bool,1>', ark_permuteDims_bool_1, 'ManipulationMsg', 389);

proc ark_permuteDims_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('permuteDims<bigint,1>', ark_permuteDims_bigint_1, 'ManipulationMsg', 389);

proc ark_reshape_int_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<int64,1,1>', ark_reshape_int_1_1, 'ManipulationMsg', 439);

proc ark_reshape_uint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<uint64,1,1>', ark_reshape_uint_1_1, 'ManipulationMsg', 439);

proc ark_reshape_uint8_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=1);
registerFunction('reshape<uint8,1,1>', ark_reshape_uint8_1_1, 'ManipulationMsg', 439);

proc ark_reshape_real_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<float64,1,1>', ark_reshape_real_1_1, 'ManipulationMsg', 439);

proc ark_reshape_bool_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<bool,1,1>', ark_reshape_bool_1_1, 'ManipulationMsg', 439);

proc ark_reshape_bigint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<bigint,1,1>', ark_reshape_bigint_1_1, 'ManipulationMsg', 439);

proc ark_roll_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('roll<int64,1>', ark_roll_int_1, 'ManipulationMsg', 521);

proc ark_roll_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('roll<uint64,1>', ark_roll_uint_1, 'ManipulationMsg', 521);

proc ark_roll_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('roll<uint8,1>', ark_roll_uint8_1, 'ManipulationMsg', 521);

proc ark_roll_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('roll<float64,1>', ark_roll_real_1, 'ManipulationMsg', 521);

proc ark_roll_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('roll<bool,1>', ark_roll_bool_1, 'ManipulationMsg', 521);

proc ark_roll_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('roll<bigint,1>', ark_roll_bigint_1, 'ManipulationMsg', 521);

proc ark_rollFlattened_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('rollFlattened<int64,1>', ark_rollFlattened_int_1, 'ManipulationMsg', 586);

proc ark_rollFlattened_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('rollFlattened<uint64,1>', ark_rollFlattened_uint_1, 'ManipulationMsg', 586);

proc ark_rollFlattened_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('rollFlattened<uint8,1>', ark_rollFlattened_uint8_1, 'ManipulationMsg', 586);

proc ark_rollFlattened_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('rollFlattened<float64,1>', ark_rollFlattened_real_1, 'ManipulationMsg', 586);

proc ark_rollFlattened_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('rollFlattened<bool,1>', ark_rollFlattened_bool_1, 'ManipulationMsg', 586);

proc ark_rollFlattened_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('rollFlattened<bigint,1>', ark_rollFlattened_bigint_1, 'ManipulationMsg', 586);

proc ark_squeeze_int_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<int64,1,1>', ark_squeeze_int_1_1, 'ManipulationMsg', 607);

proc ark_squeeze_uint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<uint64,1,1>', ark_squeeze_uint_1_1, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<uint8,1,1>', ark_squeeze_uint8_1_1, 'ManipulationMsg', 607);

proc ark_squeeze_real_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<float64,1,1>', ark_squeeze_real_1_1, 'ManipulationMsg', 607);

proc ark_squeeze_bool_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<bool,1,1>', ark_squeeze_bool_1_1, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<bigint,1,1>', ark_squeeze_bigint_1_1, 'ManipulationMsg', 607);

proc ark_stack_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('stack<int64,1>', ark_stack_int_1, 'ManipulationMsg', 686);

proc ark_stack_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('stack<uint64,1>', ark_stack_uint_1, 'ManipulationMsg', 686);

proc ark_stack_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('stack<uint8,1>', ark_stack_uint8_1, 'ManipulationMsg', 686);

proc ark_stack_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('stack<float64,1>', ark_stack_real_1, 'ManipulationMsg', 686);

proc ark_stack_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('stack<bool,1>', ark_stack_bool_1, 'ManipulationMsg', 686);

proc ark_stack_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('stack<bigint,1>', ark_stack_bigint_1, 'ManipulationMsg', 686);

proc ark_tile_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('tile<int64,1>', ark_tile_int_1, 'ManipulationMsg', 777);

proc ark_tile_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('tile<uint64,1>', ark_tile_uint_1, 'ManipulationMsg', 777);

proc ark_tile_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('tile<uint8,1>', ark_tile_uint8_1, 'ManipulationMsg', 777);

proc ark_tile_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('tile<float64,1>', ark_tile_real_1, 'ManipulationMsg', 777);

proc ark_tile_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('tile<bool,1>', ark_tile_bool_1, 'ManipulationMsg', 777);

proc ark_tile_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('tile<bigint,1>', ark_tile_bigint_1, 'ManipulationMsg', 777);

proc ark_unstack_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('unstack<int64,1>', ark_unstack_int_1, 'ManipulationMsg', 818);

proc ark_unstack_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('unstack<uint64,1>', ark_unstack_uint_1, 'ManipulationMsg', 818);

proc ark_unstack_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('unstack<uint8,1>', ark_unstack_uint8_1, 'ManipulationMsg', 818);

proc ark_unstack_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('unstack<float64,1>', ark_unstack_real_1, 'ManipulationMsg', 818);

proc ark_unstack_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('unstack<bool,1>', ark_unstack_bool_1, 'ManipulationMsg', 818);

proc ark_unstack_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('unstack<bigint,1>', ark_unstack_bigint_1, 'ManipulationMsg', 818);

proc ark_repeatFlat_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('repeatFlat<int64,1>', ark_repeatFlat_int_1, 'ManipulationMsg', 902);

proc ark_repeatFlat_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('repeatFlat<uint64,1>', ark_repeatFlat_uint_1, 'ManipulationMsg', 902);

proc ark_repeatFlat_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('repeatFlat<uint8,1>', ark_repeatFlat_uint8_1, 'ManipulationMsg', 902);

proc ark_repeatFlat_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('repeatFlat<float64,1>', ark_repeatFlat_real_1, 'ManipulationMsg', 902);

proc ark_repeatFlat_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('repeatFlat<bool,1>', ark_repeatFlat_bool_1, 'ManipulationMsg', 902);

proc ark_repeatFlat_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('repeatFlat<bigint,1>', ark_repeatFlat_bigint_1, 'ManipulationMsg', 902);

import RandMsg;

proc ark_randint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('randint<int64,1>', ark_randint_int_1, 'RandMsg', 36);

proc ark_randint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('randint<uint64,1>', ark_randint_uint_1, 'RandMsg', 36);

proc ark_randint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('randint<uint8,1>', ark_randint_uint8_1, 'RandMsg', 36);

proc ark_randint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('randint<float64,1>', ark_randint_real_1, 'RandMsg', 36);

proc ark_randint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('randint<bool,1>', ark_randint_bool_1, 'RandMsg', 36);

proc ark_randint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('randint<bigint,1>', ark_randint_bigint_1, 'RandMsg', 36);

import StatsMsg;

proc ark_reg_mean_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int): MsgTuple throws {
  var x_array_sym = st[msgArgs['x']]: SymEntry(array_dtype_0, array_nd_0);
  ref x = x_array_sym.a;
  var skipNan = msgArgs['skipNan'].toScalar(bool);
  var ark_result = StatsMsg.mean(x,skipNan);

  return MsgTuple.fromScalar(ark_result);
}

proc ark_mean_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('mean<int64,1>', ark_mean_int_1, 'StatsMsg', 22);

proc ark_mean_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('mean<uint64,1>', ark_mean_uint_1, 'StatsMsg', 22);

proc ark_mean_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('mean<uint8,1>', ark_mean_uint8_1, 'StatsMsg', 22);

proc ark_mean_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('mean<float64,1>', ark_mean_real_1, 'StatsMsg', 22);

proc ark_mean_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('mean<bool,1>', ark_mean_bool_1, 'StatsMsg', 22);

proc ark_mean_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('mean<bigint,1>', ark_mean_bigint_1, 'StatsMsg', 22);

proc ark_reg_meanReduce_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int): MsgTuple throws {
  var x_array_sym = st[msgArgs['x']]: SymEntry(array_dtype_0, array_nd_0);
  ref x = x_array_sym.a;
  var skipNan = msgArgs['skipNan'].toScalar(bool);
  var axes = msgArgs['axes'].toScalarList(int);
  var ark_result = StatsMsg.meanReduce(x,skipNan,axes);
  var ark_result_symbol = new shared SymEntry(ark_result);

  return st.insert(ark_result_symbol);
}

proc ark_meanReduce_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('meanReduce<int64,1>', ark_meanReduce_int_1, 'StatsMsg', 29);

proc ark_meanReduce_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('meanReduce<uint64,1>', ark_meanReduce_uint_1, 'StatsMsg', 29);

proc ark_meanReduce_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('meanReduce<uint8,1>', ark_meanReduce_uint8_1, 'StatsMsg', 29);

proc ark_meanReduce_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('meanReduce<float64,1>', ark_meanReduce_real_1, 'StatsMsg', 29);

proc ark_meanReduce_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('meanReduce<bool,1>', ark_meanReduce_bool_1, 'StatsMsg', 29);

proc ark_meanReduce_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('meanReduce<bigint,1>', ark_meanReduce_bigint_1, 'StatsMsg', 29);

proc ark_reg_variance_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int): MsgTuple throws {
  var x_array_sym = st[msgArgs['x']]: SymEntry(array_dtype_0, array_nd_0);
  ref x = x_array_sym.a;
  var skipNan = msgArgs['skipNan'].toScalar(bool);
  var ddof = msgArgs['ddof'].toScalar(real);
  var ark_result = StatsMsg.variance(x,skipNan,ddof);

  return MsgTuple.fromScalar(ark_result);
}

proc ark_var_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('var<int64,1>', ark_var_int_1, 'StatsMsg', 40);

proc ark_var_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('var<uint64,1>', ark_var_uint_1, 'StatsMsg', 40);

proc ark_var_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('var<uint8,1>', ark_var_uint8_1, 'StatsMsg', 40);

proc ark_var_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('var<float64,1>', ark_var_real_1, 'StatsMsg', 40);

proc ark_var_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('var<bool,1>', ark_var_bool_1, 'StatsMsg', 40);

proc ark_var_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('var<bigint,1>', ark_var_bigint_1, 'StatsMsg', 40);

proc ark_reg_varReduce_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int): MsgTuple throws {
  var x_array_sym = st[msgArgs['x']]: SymEntry(array_dtype_0, array_nd_0);
  ref x = x_array_sym.a;
  var skipNan = msgArgs['skipNan'].toScalar(bool);
  var ddof = msgArgs['ddof'].toScalar(real);
  var axes = msgArgs['axes'].toScalarList(int);
  var ark_result = StatsMsg.varReduce(x,skipNan,ddof,axes);
  var ark_result_symbol = new shared SymEntry(ark_result);

  return st.insert(ark_result_symbol);
}

proc ark_varReduce_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('varReduce<int64,1>', ark_varReduce_int_1, 'StatsMsg', 47);

proc ark_varReduce_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('varReduce<uint64,1>', ark_varReduce_uint_1, 'StatsMsg', 47);

proc ark_varReduce_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('varReduce<uint8,1>', ark_varReduce_uint8_1, 'StatsMsg', 47);

proc ark_varReduce_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('varReduce<float64,1>', ark_varReduce_real_1, 'StatsMsg', 47);

proc ark_varReduce_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('varReduce<bool,1>', ark_varReduce_bool_1, 'StatsMsg', 47);

proc ark_varReduce_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('varReduce<bigint,1>', ark_varReduce_bigint_1, 'StatsMsg', 47);

proc ark_reg_std_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int): MsgTuple throws {
  var x_array_sym = st[msgArgs['x']]: SymEntry(array_dtype_0, array_nd_0);
  ref x = x_array_sym.a;
  var skipNan = msgArgs['skipNan'].toScalar(bool);
  var ddof = msgArgs['ddof'].toScalar(real);
  var ark_result = StatsMsg.std(x,skipNan,ddof);

  return MsgTuple.fromScalar(ark_result);
}

proc ark_std_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('std<int64,1>', ark_std_int_1, 'StatsMsg', 58);

proc ark_std_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('std<uint64,1>', ark_std_uint_1, 'StatsMsg', 58);

proc ark_std_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('std<uint8,1>', ark_std_uint8_1, 'StatsMsg', 58);

proc ark_std_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('std<float64,1>', ark_std_real_1, 'StatsMsg', 58);

proc ark_std_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('std<bool,1>', ark_std_bool_1, 'StatsMsg', 58);

proc ark_std_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('std<bigint,1>', ark_std_bigint_1, 'StatsMsg', 58);

proc ark_reg_stdReduce_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int): MsgTuple throws {
  var x_array_sym = st[msgArgs['x']]: SymEntry(array_dtype_0, array_nd_0);
  ref x = x_array_sym.a;
  var skipNan = msgArgs['skipNan'].toScalar(bool);
  var ddof = msgArgs['ddof'].toScalar(real);
  var axes = msgArgs['axes'].toScalarList(int);
  var ark_result = StatsMsg.stdReduce(x,skipNan,ddof,axes);
  var ark_result_symbol = new shared SymEntry(ark_result);

  return st.insert(ark_result_symbol);
}

proc ark_stdReduce_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('stdReduce<int64,1>', ark_stdReduce_int_1, 'StatsMsg', 65);

proc ark_stdReduce_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('stdReduce<uint64,1>', ark_stdReduce_uint_1, 'StatsMsg', 65);

proc ark_stdReduce_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('stdReduce<uint8,1>', ark_stdReduce_uint8_1, 'StatsMsg', 65);

proc ark_stdReduce_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('stdReduce<float64,1>', ark_stdReduce_real_1, 'StatsMsg', 65);

proc ark_stdReduce_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('stdReduce<bool,1>', ark_stdReduce_bool_1, 'StatsMsg', 65);

proc ark_stdReduce_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('stdReduce<bigint,1>', ark_stdReduce_bigint_1, 'StatsMsg', 65);

proc ark_reg_cov_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int, type array_dtype_1, param array_nd_1: int): MsgTuple throws {
  var x_array_sym = st[msgArgs['x']]: SymEntry(array_dtype_0, array_nd_0);
  ref x = x_array_sym.a;
  var y_array_sym = st[msgArgs['y']]: SymEntry(array_dtype_1, array_nd_1);
  ref y = y_array_sym.a;
  var ark_result = StatsMsg.cov(x,y);

  return MsgTuple.fromScalar(ark_result);
}

proc ark_cov_int_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<int64,1,int64,1>', ark_cov_int_1_int_1, 'StatsMsg', 76);

proc ark_cov_int_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<int64,1,uint64,1>', ark_cov_int_1_uint_1, 'StatsMsg', 76);

proc ark_cov_int_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<int64,1,uint8,1>', ark_cov_int_1_uint8_1, 'StatsMsg', 76);

proc ark_cov_int_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<int64,1,float64,1>', ark_cov_int_1_real_1, 'StatsMsg', 76);

proc ark_cov_int_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<int64,1,bool,1>', ark_cov_int_1_bool_1, 'StatsMsg', 76);

proc ark_cov_int_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<int64,1,bigint,1>', ark_cov_int_1_bigint_1, 'StatsMsg', 76);

proc ark_cov_uint_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<uint64,1,int64,1>', ark_cov_uint_1_int_1, 'StatsMsg', 76);

proc ark_cov_uint_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<uint64,1,uint64,1>', ark_cov_uint_1_uint_1, 'StatsMsg', 76);

proc ark_cov_uint_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<uint64,1,uint8,1>', ark_cov_uint_1_uint8_1, 'StatsMsg', 76);

proc ark_cov_uint_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<uint64,1,float64,1>', ark_cov_uint_1_real_1, 'StatsMsg', 76);

proc ark_cov_uint_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<uint64,1,bool,1>', ark_cov_uint_1_bool_1, 'StatsMsg', 76);

proc ark_cov_uint_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<uint64,1,bigint,1>', ark_cov_uint_1_bigint_1, 'StatsMsg', 76);

proc ark_cov_uint8_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<uint8,1,int64,1>', ark_cov_uint8_1_int_1, 'StatsMsg', 76);

proc ark_cov_uint8_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<uint8,1,uint64,1>', ark_cov_uint8_1_uint_1, 'StatsMsg', 76);

proc ark_cov_uint8_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<uint8,1,uint8,1>', ark_cov_uint8_1_uint8_1, 'StatsMsg', 76);

proc ark_cov_uint8_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<uint8,1,float64,1>', ark_cov_uint8_1_real_1, 'StatsMsg', 76);

proc ark_cov_uint8_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<uint8,1,bool,1>', ark_cov_uint8_1_bool_1, 'StatsMsg', 76);

proc ark_cov_uint8_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<uint8,1,bigint,1>', ark_cov_uint8_1_bigint_1, 'StatsMsg', 76);

proc ark_cov_real_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<float64,1,int64,1>', ark_cov_real_1_int_1, 'StatsMsg', 76);

proc ark_cov_real_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<float64,1,uint64,1>', ark_cov_real_1_uint_1, 'StatsMsg', 76);

proc ark_cov_real_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<float64,1,uint8,1>', ark_cov_real_1_uint8_1, 'StatsMsg', 76);

proc ark_cov_real_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<float64,1,float64,1>', ark_cov_real_1_real_1, 'StatsMsg', 76);

proc ark_cov_real_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<float64,1,bool,1>', ark_cov_real_1_bool_1, 'StatsMsg', 76);

proc ark_cov_real_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<float64,1,bigint,1>', ark_cov_real_1_bigint_1, 'StatsMsg', 76);

proc ark_cov_bool_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<bool,1,int64,1>', ark_cov_bool_1_int_1, 'StatsMsg', 76);

proc ark_cov_bool_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<bool,1,uint64,1>', ark_cov_bool_1_uint_1, 'StatsMsg', 76);

proc ark_cov_bool_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<bool,1,uint8,1>', ark_cov_bool_1_uint8_1, 'StatsMsg', 76);

proc ark_cov_bool_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<bool,1,float64,1>', ark_cov_bool_1_real_1, 'StatsMsg', 76);

proc ark_cov_bool_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<bool,1,bool,1>', ark_cov_bool_1_bool_1, 'StatsMsg', 76);

proc ark_cov_bool_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<bool,1,bigint,1>', ark_cov_bool_1_bigint_1, 'StatsMsg', 76);

proc ark_cov_bigint_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<bigint,1,int64,1>', ark_cov_bigint_1_int_1, 'StatsMsg', 76);

proc ark_cov_bigint_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<bigint,1,uint64,1>', ark_cov_bigint_1_uint_1, 'StatsMsg', 76);

proc ark_cov_bigint_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<bigint,1,uint8,1>', ark_cov_bigint_1_uint8_1, 'StatsMsg', 76);

proc ark_cov_bigint_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<bigint,1,float64,1>', ark_cov_bigint_1_real_1, 'StatsMsg', 76);

proc ark_cov_bigint_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<bigint,1,bool,1>', ark_cov_bigint_1_bool_1, 'StatsMsg', 76);

proc ark_cov_bigint_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<bigint,1,bigint,1>', ark_cov_bigint_1_bigint_1, 'StatsMsg', 76);

proc ark_reg_corr_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int, type array_dtype_1, param array_nd_1: int): MsgTuple throws {
  var x_array_sym = st[msgArgs['x']]: SymEntry(array_dtype_0, array_nd_0);
  ref x = x_array_sym.a;
  var y_array_sym = st[msgArgs['y']]: SymEntry(array_dtype_1, array_nd_1);
  ref y = y_array_sym.a;
  var ark_result = StatsMsg.corr(x,y);

  return MsgTuple.fromScalar(ark_result);
}

proc ark_corr_int_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<int64,1,int64,1>', ark_corr_int_1_int_1, 'StatsMsg', 98);

proc ark_corr_int_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<int64,1,uint64,1>', ark_corr_int_1_uint_1, 'StatsMsg', 98);

proc ark_corr_int_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<int64,1,uint8,1>', ark_corr_int_1_uint8_1, 'StatsMsg', 98);

proc ark_corr_int_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<int64,1,float64,1>', ark_corr_int_1_real_1, 'StatsMsg', 98);

proc ark_corr_int_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<int64,1,bool,1>', ark_corr_int_1_bool_1, 'StatsMsg', 98);

proc ark_corr_int_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<int64,1,bigint,1>', ark_corr_int_1_bigint_1, 'StatsMsg', 98);

proc ark_corr_uint_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<uint64,1,int64,1>', ark_corr_uint_1_int_1, 'StatsMsg', 98);

proc ark_corr_uint_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<uint64,1,uint64,1>', ark_corr_uint_1_uint_1, 'StatsMsg', 98);

proc ark_corr_uint_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<uint64,1,uint8,1>', ark_corr_uint_1_uint8_1, 'StatsMsg', 98);

proc ark_corr_uint_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<uint64,1,float64,1>', ark_corr_uint_1_real_1, 'StatsMsg', 98);

proc ark_corr_uint_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<uint64,1,bool,1>', ark_corr_uint_1_bool_1, 'StatsMsg', 98);

proc ark_corr_uint_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<uint64,1,bigint,1>', ark_corr_uint_1_bigint_1, 'StatsMsg', 98);

proc ark_corr_uint8_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<uint8,1,int64,1>', ark_corr_uint8_1_int_1, 'StatsMsg', 98);

proc ark_corr_uint8_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<uint8,1,uint64,1>', ark_corr_uint8_1_uint_1, 'StatsMsg', 98);

proc ark_corr_uint8_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<uint8,1,uint8,1>', ark_corr_uint8_1_uint8_1, 'StatsMsg', 98);

proc ark_corr_uint8_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<uint8,1,float64,1>', ark_corr_uint8_1_real_1, 'StatsMsg', 98);

proc ark_corr_uint8_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<uint8,1,bool,1>', ark_corr_uint8_1_bool_1, 'StatsMsg', 98);

proc ark_corr_uint8_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<uint8,1,bigint,1>', ark_corr_uint8_1_bigint_1, 'StatsMsg', 98);

proc ark_corr_real_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<float64,1,int64,1>', ark_corr_real_1_int_1, 'StatsMsg', 98);

proc ark_corr_real_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<float64,1,uint64,1>', ark_corr_real_1_uint_1, 'StatsMsg', 98);

proc ark_corr_real_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<float64,1,uint8,1>', ark_corr_real_1_uint8_1, 'StatsMsg', 98);

proc ark_corr_real_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<float64,1,float64,1>', ark_corr_real_1_real_1, 'StatsMsg', 98);

proc ark_corr_real_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<float64,1,bool,1>', ark_corr_real_1_bool_1, 'StatsMsg', 98);

proc ark_corr_real_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<float64,1,bigint,1>', ark_corr_real_1_bigint_1, 'StatsMsg', 98);

proc ark_corr_bool_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<bool,1,int64,1>', ark_corr_bool_1_int_1, 'StatsMsg', 98);

proc ark_corr_bool_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<bool,1,uint64,1>', ark_corr_bool_1_uint_1, 'StatsMsg', 98);

proc ark_corr_bool_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<bool,1,uint8,1>', ark_corr_bool_1_uint8_1, 'StatsMsg', 98);

proc ark_corr_bool_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<bool,1,float64,1>', ark_corr_bool_1_real_1, 'StatsMsg', 98);

proc ark_corr_bool_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<bool,1,bool,1>', ark_corr_bool_1_bool_1, 'StatsMsg', 98);

proc ark_corr_bool_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<bool,1,bigint,1>', ark_corr_bool_1_bigint_1, 'StatsMsg', 98);

proc ark_corr_bigint_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<bigint,1,int64,1>', ark_corr_bigint_1_int_1, 'StatsMsg', 98);

proc ark_corr_bigint_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<bigint,1,uint64,1>', ark_corr_bigint_1_uint_1, 'StatsMsg', 98);

proc ark_corr_bigint_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<bigint,1,uint8,1>', ark_corr_bigint_1_uint8_1, 'StatsMsg', 98);

proc ark_corr_bigint_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<bigint,1,float64,1>', ark_corr_bigint_1_real_1, 'StatsMsg', 98);

proc ark_corr_bigint_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<bigint,1,bool,1>', ark_corr_bigint_1_bool_1, 'StatsMsg', 98);

proc ark_corr_bigint_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<bigint,1,bigint,1>', ark_corr_bigint_1_bigint_1, 'StatsMsg', 98);

proc ark_reg_cumSum_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int): MsgTuple throws {
  var x_array_sym = st[msgArgs['x']]: SymEntry(array_dtype_0, array_nd_0);
  ref x = x_array_sym.a;
  var axis = msgArgs['axis'].toScalar(int);
  var includeInitial = msgArgs['includeInitial'].toScalar(bool);
  var ark_result = StatsMsg.cumSum(x,axis,includeInitial);
  var ark_result_symbol = new shared SymEntry(ark_result);

  return st.insert(ark_result_symbol);
}

proc ark_cumSum_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('cumSum<int64,1>', ark_cumSum_int_1, 'StatsMsg', 120);

proc ark_cumSum_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('cumSum<uint64,1>', ark_cumSum_uint_1, 'StatsMsg', 120);

proc ark_cumSum_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('cumSum<uint8,1>', ark_cumSum_uint8_1, 'StatsMsg', 120);

proc ark_cumSum_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('cumSum<float64,1>', ark_cumSum_real_1, 'StatsMsg', 120);

proc ark_cumSum_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('cumSum<bool,1>', ark_cumSum_bool_1, 'StatsMsg', 120);

proc ark_cumSum_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('cumSum<bigint,1>', ark_cumSum_bigint_1, 'StatsMsg', 120);

import MsgProcessing;

proc ark_create_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('create<int64,1>', ark_create_int_1, 'MsgProcessing', 35);

proc ark_create_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('create<uint64,1>', ark_create_uint_1, 'MsgProcessing', 35);

proc ark_create_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('create<uint8,1>', ark_create_uint8_1, 'MsgProcessing', 35);

proc ark_create_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('create<float64,1>', ark_create_real_1, 'MsgProcessing', 35);

proc ark_create_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('create<bool,1>', ark_create_bool_1, 'MsgProcessing', 35);

proc ark_create_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('create<bigint,1>', ark_create_bigint_1, 'MsgProcessing', 35);

proc ark_createScalarArray_int(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.createScalarArray(cmd, msgArgs, st, array_dtype=int);
registerFunction('createScalarArray<int64>', ark_createScalarArray_int, 'MsgProcessing', 49);

proc ark_createScalarArray_uint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.createScalarArray(cmd, msgArgs, st, array_dtype=uint);
registerFunction('createScalarArray<uint64>', ark_createScalarArray_uint, 'MsgProcessing', 49);

proc ark_createScalarArray_uint8(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.createScalarArray(cmd, msgArgs, st, array_dtype=uint(8));
registerFunction('createScalarArray<uint8>', ark_createScalarArray_uint8, 'MsgProcessing', 49);

proc ark_createScalarArray_real(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.createScalarArray(cmd, msgArgs, st, array_dtype=real);
registerFunction('createScalarArray<float64>', ark_createScalarArray_real, 'MsgProcessing', 49);

proc ark_createScalarArray_bool(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.createScalarArray(cmd, msgArgs, st, array_dtype=bool);
registerFunction('createScalarArray<bool>', ark_createScalarArray_bool, 'MsgProcessing', 49);

proc ark_createScalarArray_bigint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.createScalarArray(cmd, msgArgs, st, array_dtype=bigint);
registerFunction('createScalarArray<bigint>', ark_createScalarArray_bigint, 'MsgProcessing', 49);

proc ark_set_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('set<int64,1>', ark_set_int_1, 'MsgProcessing', 299);

proc ark_set_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('set<uint64,1>', ark_set_uint_1, 'MsgProcessing', 299);

proc ark_set_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('set<uint8,1>', ark_set_uint8_1, 'MsgProcessing', 299);

proc ark_set_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('set<float64,1>', ark_set_real_1, 'MsgProcessing', 299);

proc ark_set_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('set<bool,1>', ark_set_bool_1, 'MsgProcessing', 299);

proc ark_set_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('set<bigint,1>', ark_set_bigint_1, 'MsgProcessing', 299);

}
