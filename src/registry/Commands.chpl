module Commands {

use CommandMap, Message, MultiTypeSymbolTable, MultiTypeSymEntry;

use BigInteger;

param regConfig = """
{
  "parameter_classes": {
    "array": {
      "nd": [
        1,
        2,
        3,
        4
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

proc ark_cast_int_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=int, array_nd=2);
registerFunction('cast<int64,int64,2>', ark_cast_int_int_2, 'CastMsg', 23);

proc ark_cast_int_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=int, array_nd=3);
registerFunction('cast<int64,int64,3>', ark_cast_int_int_3, 'CastMsg', 23);

proc ark_cast_int_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=int, array_nd=4);
registerFunction('cast<int64,int64,4>', ark_cast_int_int_4, 'CastMsg', 23);

proc ark_cast_int_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=uint, array_nd=1);
registerFunction('cast<int64,uint64,1>', ark_cast_int_uint_1, 'CastMsg', 23);

proc ark_cast_int_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=uint, array_nd=2);
registerFunction('cast<int64,uint64,2>', ark_cast_int_uint_2, 'CastMsg', 23);

proc ark_cast_int_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=uint, array_nd=3);
registerFunction('cast<int64,uint64,3>', ark_cast_int_uint_3, 'CastMsg', 23);

proc ark_cast_int_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=uint, array_nd=4);
registerFunction('cast<int64,uint64,4>', ark_cast_int_uint_4, 'CastMsg', 23);

proc ark_cast_int_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=uint(8), array_nd=1);
registerFunction('cast<int64,uint8,1>', ark_cast_int_uint8_1, 'CastMsg', 23);

proc ark_cast_int_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=uint(8), array_nd=2);
registerFunction('cast<int64,uint8,2>', ark_cast_int_uint8_2, 'CastMsg', 23);

proc ark_cast_int_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=uint(8), array_nd=3);
registerFunction('cast<int64,uint8,3>', ark_cast_int_uint8_3, 'CastMsg', 23);

proc ark_cast_int_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=uint(8), array_nd=4);
registerFunction('cast<int64,uint8,4>', ark_cast_int_uint8_4, 'CastMsg', 23);

proc ark_cast_int_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=real, array_nd=1);
registerFunction('cast<int64,float64,1>', ark_cast_int_real_1, 'CastMsg', 23);

proc ark_cast_int_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=real, array_nd=2);
registerFunction('cast<int64,float64,2>', ark_cast_int_real_2, 'CastMsg', 23);

proc ark_cast_int_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=real, array_nd=3);
registerFunction('cast<int64,float64,3>', ark_cast_int_real_3, 'CastMsg', 23);

proc ark_cast_int_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=real, array_nd=4);
registerFunction('cast<int64,float64,4>', ark_cast_int_real_4, 'CastMsg', 23);

proc ark_cast_int_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=bool, array_nd=1);
registerFunction('cast<int64,bool,1>', ark_cast_int_bool_1, 'CastMsg', 23);

proc ark_cast_int_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=bool, array_nd=2);
registerFunction('cast<int64,bool,2>', ark_cast_int_bool_2, 'CastMsg', 23);

proc ark_cast_int_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=bool, array_nd=3);
registerFunction('cast<int64,bool,3>', ark_cast_int_bool_3, 'CastMsg', 23);

proc ark_cast_int_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=bool, array_nd=4);
registerFunction('cast<int64,bool,4>', ark_cast_int_bool_4, 'CastMsg', 23);

proc ark_cast_int_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=bigint, array_nd=1);
registerFunction('cast<int64,bigint,1>', ark_cast_int_bigint_1, 'CastMsg', 23);

proc ark_cast_int_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=bigint, array_nd=2);
registerFunction('cast<int64,bigint,2>', ark_cast_int_bigint_2, 'CastMsg', 23);

proc ark_cast_int_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=bigint, array_nd=3);
registerFunction('cast<int64,bigint,3>', ark_cast_int_bigint_3, 'CastMsg', 23);

proc ark_cast_int_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=int, array_dtype_to=bigint, array_nd=4);
registerFunction('cast<int64,bigint,4>', ark_cast_int_bigint_4, 'CastMsg', 23);

proc ark_cast_uint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=int, array_nd=1);
registerFunction('cast<uint64,int64,1>', ark_cast_uint_int_1, 'CastMsg', 23);

proc ark_cast_uint_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=int, array_nd=2);
registerFunction('cast<uint64,int64,2>', ark_cast_uint_int_2, 'CastMsg', 23);

proc ark_cast_uint_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=int, array_nd=3);
registerFunction('cast<uint64,int64,3>', ark_cast_uint_int_3, 'CastMsg', 23);

proc ark_cast_uint_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=int, array_nd=4);
registerFunction('cast<uint64,int64,4>', ark_cast_uint_int_4, 'CastMsg', 23);

proc ark_cast_uint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=uint, array_nd=1);
registerFunction('cast<uint64,uint64,1>', ark_cast_uint_uint_1, 'CastMsg', 23);

proc ark_cast_uint_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=uint, array_nd=2);
registerFunction('cast<uint64,uint64,2>', ark_cast_uint_uint_2, 'CastMsg', 23);

proc ark_cast_uint_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=uint, array_nd=3);
registerFunction('cast<uint64,uint64,3>', ark_cast_uint_uint_3, 'CastMsg', 23);

proc ark_cast_uint_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=uint, array_nd=4);
registerFunction('cast<uint64,uint64,4>', ark_cast_uint_uint_4, 'CastMsg', 23);

proc ark_cast_uint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=uint(8), array_nd=1);
registerFunction('cast<uint64,uint8,1>', ark_cast_uint_uint8_1, 'CastMsg', 23);

proc ark_cast_uint_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=uint(8), array_nd=2);
registerFunction('cast<uint64,uint8,2>', ark_cast_uint_uint8_2, 'CastMsg', 23);

proc ark_cast_uint_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=uint(8), array_nd=3);
registerFunction('cast<uint64,uint8,3>', ark_cast_uint_uint8_3, 'CastMsg', 23);

proc ark_cast_uint_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=uint(8), array_nd=4);
registerFunction('cast<uint64,uint8,4>', ark_cast_uint_uint8_4, 'CastMsg', 23);

proc ark_cast_uint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=real, array_nd=1);
registerFunction('cast<uint64,float64,1>', ark_cast_uint_real_1, 'CastMsg', 23);

proc ark_cast_uint_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=real, array_nd=2);
registerFunction('cast<uint64,float64,2>', ark_cast_uint_real_2, 'CastMsg', 23);

proc ark_cast_uint_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=real, array_nd=3);
registerFunction('cast<uint64,float64,3>', ark_cast_uint_real_3, 'CastMsg', 23);

proc ark_cast_uint_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=real, array_nd=4);
registerFunction('cast<uint64,float64,4>', ark_cast_uint_real_4, 'CastMsg', 23);

proc ark_cast_uint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=bool, array_nd=1);
registerFunction('cast<uint64,bool,1>', ark_cast_uint_bool_1, 'CastMsg', 23);

proc ark_cast_uint_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=bool, array_nd=2);
registerFunction('cast<uint64,bool,2>', ark_cast_uint_bool_2, 'CastMsg', 23);

proc ark_cast_uint_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=bool, array_nd=3);
registerFunction('cast<uint64,bool,3>', ark_cast_uint_bool_3, 'CastMsg', 23);

proc ark_cast_uint_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=bool, array_nd=4);
registerFunction('cast<uint64,bool,4>', ark_cast_uint_bool_4, 'CastMsg', 23);

proc ark_cast_uint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=bigint, array_nd=1);
registerFunction('cast<uint64,bigint,1>', ark_cast_uint_bigint_1, 'CastMsg', 23);

proc ark_cast_uint_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=bigint, array_nd=2);
registerFunction('cast<uint64,bigint,2>', ark_cast_uint_bigint_2, 'CastMsg', 23);

proc ark_cast_uint_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=bigint, array_nd=3);
registerFunction('cast<uint64,bigint,3>', ark_cast_uint_bigint_3, 'CastMsg', 23);

proc ark_cast_uint_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint, array_dtype_to=bigint, array_nd=4);
registerFunction('cast<uint64,bigint,4>', ark_cast_uint_bigint_4, 'CastMsg', 23);

proc ark_cast_uint8_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=int, array_nd=1);
registerFunction('cast<uint8,int64,1>', ark_cast_uint8_int_1, 'CastMsg', 23);

proc ark_cast_uint8_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=int, array_nd=2);
registerFunction('cast<uint8,int64,2>', ark_cast_uint8_int_2, 'CastMsg', 23);

proc ark_cast_uint8_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=int, array_nd=3);
registerFunction('cast<uint8,int64,3>', ark_cast_uint8_int_3, 'CastMsg', 23);

proc ark_cast_uint8_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=int, array_nd=4);
registerFunction('cast<uint8,int64,4>', ark_cast_uint8_int_4, 'CastMsg', 23);

proc ark_cast_uint8_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=uint, array_nd=1);
registerFunction('cast<uint8,uint64,1>', ark_cast_uint8_uint_1, 'CastMsg', 23);

proc ark_cast_uint8_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=uint, array_nd=2);
registerFunction('cast<uint8,uint64,2>', ark_cast_uint8_uint_2, 'CastMsg', 23);

proc ark_cast_uint8_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=uint, array_nd=3);
registerFunction('cast<uint8,uint64,3>', ark_cast_uint8_uint_3, 'CastMsg', 23);

proc ark_cast_uint8_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=uint, array_nd=4);
registerFunction('cast<uint8,uint64,4>', ark_cast_uint8_uint_4, 'CastMsg', 23);

proc ark_cast_uint8_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=uint(8), array_nd=1);
registerFunction('cast<uint8,uint8,1>', ark_cast_uint8_uint8_1, 'CastMsg', 23);

proc ark_cast_uint8_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=uint(8), array_nd=2);
registerFunction('cast<uint8,uint8,2>', ark_cast_uint8_uint8_2, 'CastMsg', 23);

proc ark_cast_uint8_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=uint(8), array_nd=3);
registerFunction('cast<uint8,uint8,3>', ark_cast_uint8_uint8_3, 'CastMsg', 23);

proc ark_cast_uint8_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=uint(8), array_nd=4);
registerFunction('cast<uint8,uint8,4>', ark_cast_uint8_uint8_4, 'CastMsg', 23);

proc ark_cast_uint8_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=real, array_nd=1);
registerFunction('cast<uint8,float64,1>', ark_cast_uint8_real_1, 'CastMsg', 23);

proc ark_cast_uint8_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=real, array_nd=2);
registerFunction('cast<uint8,float64,2>', ark_cast_uint8_real_2, 'CastMsg', 23);

proc ark_cast_uint8_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=real, array_nd=3);
registerFunction('cast<uint8,float64,3>', ark_cast_uint8_real_3, 'CastMsg', 23);

proc ark_cast_uint8_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=real, array_nd=4);
registerFunction('cast<uint8,float64,4>', ark_cast_uint8_real_4, 'CastMsg', 23);

proc ark_cast_uint8_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=bool, array_nd=1);
registerFunction('cast<uint8,bool,1>', ark_cast_uint8_bool_1, 'CastMsg', 23);

proc ark_cast_uint8_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=bool, array_nd=2);
registerFunction('cast<uint8,bool,2>', ark_cast_uint8_bool_2, 'CastMsg', 23);

proc ark_cast_uint8_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=bool, array_nd=3);
registerFunction('cast<uint8,bool,3>', ark_cast_uint8_bool_3, 'CastMsg', 23);

proc ark_cast_uint8_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=bool, array_nd=4);
registerFunction('cast<uint8,bool,4>', ark_cast_uint8_bool_4, 'CastMsg', 23);

proc ark_cast_uint8_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=bigint, array_nd=1);
registerFunction('cast<uint8,bigint,1>', ark_cast_uint8_bigint_1, 'CastMsg', 23);

proc ark_cast_uint8_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=bigint, array_nd=2);
registerFunction('cast<uint8,bigint,2>', ark_cast_uint8_bigint_2, 'CastMsg', 23);

proc ark_cast_uint8_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=bigint, array_nd=3);
registerFunction('cast<uint8,bigint,3>', ark_cast_uint8_bigint_3, 'CastMsg', 23);

proc ark_cast_uint8_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=uint(8), array_dtype_to=bigint, array_nd=4);
registerFunction('cast<uint8,bigint,4>', ark_cast_uint8_bigint_4, 'CastMsg', 23);

proc ark_cast_real_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=int, array_nd=1);
registerFunction('cast<float64,int64,1>', ark_cast_real_int_1, 'CastMsg', 23);

proc ark_cast_real_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=int, array_nd=2);
registerFunction('cast<float64,int64,2>', ark_cast_real_int_2, 'CastMsg', 23);

proc ark_cast_real_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=int, array_nd=3);
registerFunction('cast<float64,int64,3>', ark_cast_real_int_3, 'CastMsg', 23);

proc ark_cast_real_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=int, array_nd=4);
registerFunction('cast<float64,int64,4>', ark_cast_real_int_4, 'CastMsg', 23);

proc ark_cast_real_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=uint, array_nd=1);
registerFunction('cast<float64,uint64,1>', ark_cast_real_uint_1, 'CastMsg', 23);

proc ark_cast_real_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=uint, array_nd=2);
registerFunction('cast<float64,uint64,2>', ark_cast_real_uint_2, 'CastMsg', 23);

proc ark_cast_real_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=uint, array_nd=3);
registerFunction('cast<float64,uint64,3>', ark_cast_real_uint_3, 'CastMsg', 23);

proc ark_cast_real_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=uint, array_nd=4);
registerFunction('cast<float64,uint64,4>', ark_cast_real_uint_4, 'CastMsg', 23);

proc ark_cast_real_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=uint(8), array_nd=1);
registerFunction('cast<float64,uint8,1>', ark_cast_real_uint8_1, 'CastMsg', 23);

proc ark_cast_real_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=uint(8), array_nd=2);
registerFunction('cast<float64,uint8,2>', ark_cast_real_uint8_2, 'CastMsg', 23);

proc ark_cast_real_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=uint(8), array_nd=3);
registerFunction('cast<float64,uint8,3>', ark_cast_real_uint8_3, 'CastMsg', 23);

proc ark_cast_real_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=uint(8), array_nd=4);
registerFunction('cast<float64,uint8,4>', ark_cast_real_uint8_4, 'CastMsg', 23);

proc ark_cast_real_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=real, array_nd=1);
registerFunction('cast<float64,float64,1>', ark_cast_real_real_1, 'CastMsg', 23);

proc ark_cast_real_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=real, array_nd=2);
registerFunction('cast<float64,float64,2>', ark_cast_real_real_2, 'CastMsg', 23);

proc ark_cast_real_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=real, array_nd=3);
registerFunction('cast<float64,float64,3>', ark_cast_real_real_3, 'CastMsg', 23);

proc ark_cast_real_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=real, array_nd=4);
registerFunction('cast<float64,float64,4>', ark_cast_real_real_4, 'CastMsg', 23);

proc ark_cast_real_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=bool, array_nd=1);
registerFunction('cast<float64,bool,1>', ark_cast_real_bool_1, 'CastMsg', 23);

proc ark_cast_real_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=bool, array_nd=2);
registerFunction('cast<float64,bool,2>', ark_cast_real_bool_2, 'CastMsg', 23);

proc ark_cast_real_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=bool, array_nd=3);
registerFunction('cast<float64,bool,3>', ark_cast_real_bool_3, 'CastMsg', 23);

proc ark_cast_real_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=bool, array_nd=4);
registerFunction('cast<float64,bool,4>', ark_cast_real_bool_4, 'CastMsg', 23);

proc ark_cast_real_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=bigint, array_nd=1);
registerFunction('cast<float64,bigint,1>', ark_cast_real_bigint_1, 'CastMsg', 23);

proc ark_cast_real_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=bigint, array_nd=2);
registerFunction('cast<float64,bigint,2>', ark_cast_real_bigint_2, 'CastMsg', 23);

proc ark_cast_real_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=bigint, array_nd=3);
registerFunction('cast<float64,bigint,3>', ark_cast_real_bigint_3, 'CastMsg', 23);

proc ark_cast_real_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=real, array_dtype_to=bigint, array_nd=4);
registerFunction('cast<float64,bigint,4>', ark_cast_real_bigint_4, 'CastMsg', 23);

proc ark_cast_bool_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=int, array_nd=1);
registerFunction('cast<bool,int64,1>', ark_cast_bool_int_1, 'CastMsg', 23);

proc ark_cast_bool_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=int, array_nd=2);
registerFunction('cast<bool,int64,2>', ark_cast_bool_int_2, 'CastMsg', 23);

proc ark_cast_bool_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=int, array_nd=3);
registerFunction('cast<bool,int64,3>', ark_cast_bool_int_3, 'CastMsg', 23);

proc ark_cast_bool_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=int, array_nd=4);
registerFunction('cast<bool,int64,4>', ark_cast_bool_int_4, 'CastMsg', 23);

proc ark_cast_bool_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=uint, array_nd=1);
registerFunction('cast<bool,uint64,1>', ark_cast_bool_uint_1, 'CastMsg', 23);

proc ark_cast_bool_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=uint, array_nd=2);
registerFunction('cast<bool,uint64,2>', ark_cast_bool_uint_2, 'CastMsg', 23);

proc ark_cast_bool_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=uint, array_nd=3);
registerFunction('cast<bool,uint64,3>', ark_cast_bool_uint_3, 'CastMsg', 23);

proc ark_cast_bool_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=uint, array_nd=4);
registerFunction('cast<bool,uint64,4>', ark_cast_bool_uint_4, 'CastMsg', 23);

proc ark_cast_bool_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=uint(8), array_nd=1);
registerFunction('cast<bool,uint8,1>', ark_cast_bool_uint8_1, 'CastMsg', 23);

proc ark_cast_bool_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=uint(8), array_nd=2);
registerFunction('cast<bool,uint8,2>', ark_cast_bool_uint8_2, 'CastMsg', 23);

proc ark_cast_bool_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=uint(8), array_nd=3);
registerFunction('cast<bool,uint8,3>', ark_cast_bool_uint8_3, 'CastMsg', 23);

proc ark_cast_bool_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=uint(8), array_nd=4);
registerFunction('cast<bool,uint8,4>', ark_cast_bool_uint8_4, 'CastMsg', 23);

proc ark_cast_bool_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=real, array_nd=1);
registerFunction('cast<bool,float64,1>', ark_cast_bool_real_1, 'CastMsg', 23);

proc ark_cast_bool_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=real, array_nd=2);
registerFunction('cast<bool,float64,2>', ark_cast_bool_real_2, 'CastMsg', 23);

proc ark_cast_bool_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=real, array_nd=3);
registerFunction('cast<bool,float64,3>', ark_cast_bool_real_3, 'CastMsg', 23);

proc ark_cast_bool_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=real, array_nd=4);
registerFunction('cast<bool,float64,4>', ark_cast_bool_real_4, 'CastMsg', 23);

proc ark_cast_bool_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=bool, array_nd=1);
registerFunction('cast<bool,bool,1>', ark_cast_bool_bool_1, 'CastMsg', 23);

proc ark_cast_bool_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=bool, array_nd=2);
registerFunction('cast<bool,bool,2>', ark_cast_bool_bool_2, 'CastMsg', 23);

proc ark_cast_bool_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=bool, array_nd=3);
registerFunction('cast<bool,bool,3>', ark_cast_bool_bool_3, 'CastMsg', 23);

proc ark_cast_bool_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=bool, array_nd=4);
registerFunction('cast<bool,bool,4>', ark_cast_bool_bool_4, 'CastMsg', 23);

proc ark_cast_bool_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=bigint, array_nd=1);
registerFunction('cast<bool,bigint,1>', ark_cast_bool_bigint_1, 'CastMsg', 23);

proc ark_cast_bool_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=bigint, array_nd=2);
registerFunction('cast<bool,bigint,2>', ark_cast_bool_bigint_2, 'CastMsg', 23);

proc ark_cast_bool_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=bigint, array_nd=3);
registerFunction('cast<bool,bigint,3>', ark_cast_bool_bigint_3, 'CastMsg', 23);

proc ark_cast_bool_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bool, array_dtype_to=bigint, array_nd=4);
registerFunction('cast<bool,bigint,4>', ark_cast_bool_bigint_4, 'CastMsg', 23);

proc ark_cast_bigint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=int, array_nd=1);
registerFunction('cast<bigint,int64,1>', ark_cast_bigint_int_1, 'CastMsg', 23);

proc ark_cast_bigint_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=int, array_nd=2);
registerFunction('cast<bigint,int64,2>', ark_cast_bigint_int_2, 'CastMsg', 23);

proc ark_cast_bigint_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=int, array_nd=3);
registerFunction('cast<bigint,int64,3>', ark_cast_bigint_int_3, 'CastMsg', 23);

proc ark_cast_bigint_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=int, array_nd=4);
registerFunction('cast<bigint,int64,4>', ark_cast_bigint_int_4, 'CastMsg', 23);

proc ark_cast_bigint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=uint, array_nd=1);
registerFunction('cast<bigint,uint64,1>', ark_cast_bigint_uint_1, 'CastMsg', 23);

proc ark_cast_bigint_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=uint, array_nd=2);
registerFunction('cast<bigint,uint64,2>', ark_cast_bigint_uint_2, 'CastMsg', 23);

proc ark_cast_bigint_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=uint, array_nd=3);
registerFunction('cast<bigint,uint64,3>', ark_cast_bigint_uint_3, 'CastMsg', 23);

proc ark_cast_bigint_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=uint, array_nd=4);
registerFunction('cast<bigint,uint64,4>', ark_cast_bigint_uint_4, 'CastMsg', 23);

proc ark_cast_bigint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=uint(8), array_nd=1);
registerFunction('cast<bigint,uint8,1>', ark_cast_bigint_uint8_1, 'CastMsg', 23);

proc ark_cast_bigint_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=uint(8), array_nd=2);
registerFunction('cast<bigint,uint8,2>', ark_cast_bigint_uint8_2, 'CastMsg', 23);

proc ark_cast_bigint_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=uint(8), array_nd=3);
registerFunction('cast<bigint,uint8,3>', ark_cast_bigint_uint8_3, 'CastMsg', 23);

proc ark_cast_bigint_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=uint(8), array_nd=4);
registerFunction('cast<bigint,uint8,4>', ark_cast_bigint_uint8_4, 'CastMsg', 23);

proc ark_cast_bigint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=real, array_nd=1);
registerFunction('cast<bigint,float64,1>', ark_cast_bigint_real_1, 'CastMsg', 23);

proc ark_cast_bigint_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=real, array_nd=2);
registerFunction('cast<bigint,float64,2>', ark_cast_bigint_real_2, 'CastMsg', 23);

proc ark_cast_bigint_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=real, array_nd=3);
registerFunction('cast<bigint,float64,3>', ark_cast_bigint_real_3, 'CastMsg', 23);

proc ark_cast_bigint_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=real, array_nd=4);
registerFunction('cast<bigint,float64,4>', ark_cast_bigint_real_4, 'CastMsg', 23);

proc ark_cast_bigint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=bool, array_nd=1);
registerFunction('cast<bigint,bool,1>', ark_cast_bigint_bool_1, 'CastMsg', 23);

proc ark_cast_bigint_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=bool, array_nd=2);
registerFunction('cast<bigint,bool,2>', ark_cast_bigint_bool_2, 'CastMsg', 23);

proc ark_cast_bigint_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=bool, array_nd=3);
registerFunction('cast<bigint,bool,3>', ark_cast_bigint_bool_3, 'CastMsg', 23);

proc ark_cast_bigint_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=bool, array_nd=4);
registerFunction('cast<bigint,bool,4>', ark_cast_bigint_bool_4, 'CastMsg', 23);

proc ark_cast_bigint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=bigint, array_nd=1);
registerFunction('cast<bigint,bigint,1>', ark_cast_bigint_bigint_1, 'CastMsg', 23);

proc ark_cast_bigint_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=bigint, array_nd=2);
registerFunction('cast<bigint,bigint,2>', ark_cast_bigint_bigint_2, 'CastMsg', 23);

proc ark_cast_bigint_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=bigint, array_nd=3);
registerFunction('cast<bigint,bigint,3>', ark_cast_bigint_bigint_3, 'CastMsg', 23);

proc ark_cast_bigint_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return CastMsg.castArray(cmd, msgArgs, st, array_dtype_from=bigint, array_dtype_to=bigint, array_nd=4);
registerFunction('cast<bigint,bigint,4>', ark_cast_bigint_bigint_4, 'CastMsg', 23);

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
  var array_array_sym = st[msgArgs['array']]: SymEntry(array_dtype_0, array_nd_0);
  ref array = array_array_sym.a;
  var idx = msgArgs['idx'].toScalarTuple(int, array_nd_0);
  var ark_result = IndexingMsg.intIndex(array,idx);

  return MsgTuple.fromScalar(ark_result);
}

proc ark__int__int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('[int]<int64,1>', ark__int__int_1, 'IndexingMsg', 194);

proc ark__int__int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2);
registerFunction('[int]<int64,2>', ark__int__int_2, 'IndexingMsg', 194);

proc ark__int__int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3);
registerFunction('[int]<int64,3>', ark__int__int_3, 'IndexingMsg', 194);

proc ark__int__int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4);
registerFunction('[int]<int64,4>', ark__int__int_4, 'IndexingMsg', 194);

proc ark__int__uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('[int]<uint64,1>', ark__int__uint_1, 'IndexingMsg', 194);

proc ark__int__uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2);
registerFunction('[int]<uint64,2>', ark__int__uint_2, 'IndexingMsg', 194);

proc ark__int__uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3);
registerFunction('[int]<uint64,3>', ark__int__uint_3, 'IndexingMsg', 194);

proc ark__int__uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4);
registerFunction('[int]<uint64,4>', ark__int__uint_4, 'IndexingMsg', 194);

proc ark__int__uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('[int]<uint8,1>', ark__int__uint8_1, 'IndexingMsg', 194);

proc ark__int__uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2);
registerFunction('[int]<uint8,2>', ark__int__uint8_2, 'IndexingMsg', 194);

proc ark__int__uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3);
registerFunction('[int]<uint8,3>', ark__int__uint8_3, 'IndexingMsg', 194);

proc ark__int__uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4);
registerFunction('[int]<uint8,4>', ark__int__uint8_4, 'IndexingMsg', 194);

proc ark__int__real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('[int]<float64,1>', ark__int__real_1, 'IndexingMsg', 194);

proc ark__int__real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2);
registerFunction('[int]<float64,2>', ark__int__real_2, 'IndexingMsg', 194);

proc ark__int__real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3);
registerFunction('[int]<float64,3>', ark__int__real_3, 'IndexingMsg', 194);

proc ark__int__real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4);
registerFunction('[int]<float64,4>', ark__int__real_4, 'IndexingMsg', 194);

proc ark__int__bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('[int]<bool,1>', ark__int__bool_1, 'IndexingMsg', 194);

proc ark__int__bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2);
registerFunction('[int]<bool,2>', ark__int__bool_2, 'IndexingMsg', 194);

proc ark__int__bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3);
registerFunction('[int]<bool,3>', ark__int__bool_3, 'IndexingMsg', 194);

proc ark__int__bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4);
registerFunction('[int]<bool,4>', ark__int__bool_4, 'IndexingMsg', 194);

proc ark__int__bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('[int]<bigint,1>', ark__int__bigint_1, 'IndexingMsg', 194);

proc ark__int__bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2);
registerFunction('[int]<bigint,2>', ark__int__bigint_2, 'IndexingMsg', 194);

proc ark__int__bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3);
registerFunction('[int]<bigint,3>', ark__int__bigint_3, 'IndexingMsg', 194);

proc ark__int__bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_intIndex_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4);
registerFunction('[int]<bigint,4>', ark__int__bigint_4, 'IndexingMsg', 194);

proc ark_reg_sliceIndex_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int): MsgTuple throws {
  var array_array_sym = st[msgArgs['array']]: SymEntry(array_dtype_0, array_nd_0);
  ref array = array_array_sym.a;
  var starts = msgArgs['starts'].toScalarTuple(int, array_nd_0);
  var stops = msgArgs['stops'].toScalarTuple(int, array_nd_0);
  var strides = msgArgs['strides'].toScalarTuple(int, array_nd_0);
  var max_bits = msgArgs['max_bits'].toScalar(int);
  var ark_result = IndexingMsg.sliceIndex(array,starts,stops,strides,max_bits);

  return st.insert(ark_result);
}

proc ark__slice__int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('[slice]<int64,1>', ark__slice__int_1, 'IndexingMsg', 211);

proc ark__slice__int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2);
registerFunction('[slice]<int64,2>', ark__slice__int_2, 'IndexingMsg', 211);

proc ark__slice__int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3);
registerFunction('[slice]<int64,3>', ark__slice__int_3, 'IndexingMsg', 211);

proc ark__slice__int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4);
registerFunction('[slice]<int64,4>', ark__slice__int_4, 'IndexingMsg', 211);

proc ark__slice__uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('[slice]<uint64,1>', ark__slice__uint_1, 'IndexingMsg', 211);

proc ark__slice__uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2);
registerFunction('[slice]<uint64,2>', ark__slice__uint_2, 'IndexingMsg', 211);

proc ark__slice__uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3);
registerFunction('[slice]<uint64,3>', ark__slice__uint_3, 'IndexingMsg', 211);

proc ark__slice__uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4);
registerFunction('[slice]<uint64,4>', ark__slice__uint_4, 'IndexingMsg', 211);

proc ark__slice__uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('[slice]<uint8,1>', ark__slice__uint8_1, 'IndexingMsg', 211);

proc ark__slice__uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2);
registerFunction('[slice]<uint8,2>', ark__slice__uint8_2, 'IndexingMsg', 211);

proc ark__slice__uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3);
registerFunction('[slice]<uint8,3>', ark__slice__uint8_3, 'IndexingMsg', 211);

proc ark__slice__uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4);
registerFunction('[slice]<uint8,4>', ark__slice__uint8_4, 'IndexingMsg', 211);

proc ark__slice__real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('[slice]<float64,1>', ark__slice__real_1, 'IndexingMsg', 211);

proc ark__slice__real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2);
registerFunction('[slice]<float64,2>', ark__slice__real_2, 'IndexingMsg', 211);

proc ark__slice__real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3);
registerFunction('[slice]<float64,3>', ark__slice__real_3, 'IndexingMsg', 211);

proc ark__slice__real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4);
registerFunction('[slice]<float64,4>', ark__slice__real_4, 'IndexingMsg', 211);

proc ark__slice__bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('[slice]<bool,1>', ark__slice__bool_1, 'IndexingMsg', 211);

proc ark__slice__bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2);
registerFunction('[slice]<bool,2>', ark__slice__bool_2, 'IndexingMsg', 211);

proc ark__slice__bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3);
registerFunction('[slice]<bool,3>', ark__slice__bool_3, 'IndexingMsg', 211);

proc ark__slice__bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4);
registerFunction('[slice]<bool,4>', ark__slice__bool_4, 'IndexingMsg', 211);

proc ark__slice__bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('[slice]<bigint,1>', ark__slice__bigint_1, 'IndexingMsg', 211);

proc ark__slice__bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2);
registerFunction('[slice]<bigint,2>', ark__slice__bigint_2, 'IndexingMsg', 211);

proc ark__slice__bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3);
registerFunction('[slice]<bigint,3>', ark__slice__bigint_3, 'IndexingMsg', 211);

proc ark__slice__bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4);
registerFunction('[slice]<bigint,4>', ark__slice__bigint_4, 'IndexingMsg', 211);

proc ark_reg_setIndexToValue_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int): MsgTuple throws {
  var array_array_sym = st[msgArgs['array']]: SymEntry(array_dtype_0, array_nd_0);
  ref array = array_array_sym.a;
  var idx = msgArgs['idx'].toScalarTuple(int, array_nd_0);
  var value = msgArgs['value'].toScalar(array_dtype_0);
  var max_bits = msgArgs['max_bits'].toScalar(int);
  IndexingMsg.setIndexToValue(array,idx,value,max_bits);

  return MsgTuple.success();
}

proc ark__int__val_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('[int]=val<int64,1>', ark__int__val_int_1, 'IndexingMsg', 504);

proc ark__int__val_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2);
registerFunction('[int]=val<int64,2>', ark__int__val_int_2, 'IndexingMsg', 504);

proc ark__int__val_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3);
registerFunction('[int]=val<int64,3>', ark__int__val_int_3, 'IndexingMsg', 504);

proc ark__int__val_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4);
registerFunction('[int]=val<int64,4>', ark__int__val_int_4, 'IndexingMsg', 504);

proc ark__int__val_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('[int]=val<uint64,1>', ark__int__val_uint_1, 'IndexingMsg', 504);

proc ark__int__val_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2);
registerFunction('[int]=val<uint64,2>', ark__int__val_uint_2, 'IndexingMsg', 504);

proc ark__int__val_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3);
registerFunction('[int]=val<uint64,3>', ark__int__val_uint_3, 'IndexingMsg', 504);

proc ark__int__val_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4);
registerFunction('[int]=val<uint64,4>', ark__int__val_uint_4, 'IndexingMsg', 504);

proc ark__int__val_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('[int]=val<uint8,1>', ark__int__val_uint8_1, 'IndexingMsg', 504);

proc ark__int__val_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2);
registerFunction('[int]=val<uint8,2>', ark__int__val_uint8_2, 'IndexingMsg', 504);

proc ark__int__val_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3);
registerFunction('[int]=val<uint8,3>', ark__int__val_uint8_3, 'IndexingMsg', 504);

proc ark__int__val_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4);
registerFunction('[int]=val<uint8,4>', ark__int__val_uint8_4, 'IndexingMsg', 504);

proc ark__int__val_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('[int]=val<float64,1>', ark__int__val_real_1, 'IndexingMsg', 504);

proc ark__int__val_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2);
registerFunction('[int]=val<float64,2>', ark__int__val_real_2, 'IndexingMsg', 504);

proc ark__int__val_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3);
registerFunction('[int]=val<float64,3>', ark__int__val_real_3, 'IndexingMsg', 504);

proc ark__int__val_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4);
registerFunction('[int]=val<float64,4>', ark__int__val_real_4, 'IndexingMsg', 504);

proc ark__int__val_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('[int]=val<bool,1>', ark__int__val_bool_1, 'IndexingMsg', 504);

proc ark__int__val_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2);
registerFunction('[int]=val<bool,2>', ark__int__val_bool_2, 'IndexingMsg', 504);

proc ark__int__val_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3);
registerFunction('[int]=val<bool,3>', ark__int__val_bool_3, 'IndexingMsg', 504);

proc ark__int__val_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4);
registerFunction('[int]=val<bool,4>', ark__int__val_bool_4, 'IndexingMsg', 504);

proc ark__int__val_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('[int]=val<bigint,1>', ark__int__val_bigint_1, 'IndexingMsg', 504);

proc ark__int__val_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2);
registerFunction('[int]=val<bigint,2>', ark__int__val_bigint_2, 'IndexingMsg', 504);

proc ark__int__val_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3);
registerFunction('[int]=val<bigint,3>', ark__int__val_bigint_3, 'IndexingMsg', 504);

proc ark__int__val_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4);
registerFunction('[int]=val<bigint,4>', ark__int__val_bigint_4, 'IndexingMsg', 504);

proc ark_reg_setSliceIndexToValue_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int): MsgTuple throws {
  var array_array_sym = st[msgArgs['array']]: SymEntry(array_dtype_0, array_nd_0);
  ref array = array_array_sym.a;
  var starts = msgArgs['starts'].toScalarTuple(int, array_nd_0);
  var stops = msgArgs['stops'].toScalarTuple(int, array_nd_0);
  var strides = msgArgs['strides'].toScalarTuple(int, array_nd_0);
  var value = msgArgs['value'].toScalar(array_dtype_0);
  var max_bits = msgArgs['max_bits'].toScalar(int);
  IndexingMsg.setSliceIndexToValue(array,starts,stops,strides,value,max_bits);

  return MsgTuple.success();
}

proc ark__slice__val_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('[slice]=val<int64,1>', ark__slice__val_int_1, 'IndexingMsg', 916);

proc ark__slice__val_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2);
registerFunction('[slice]=val<int64,2>', ark__slice__val_int_2, 'IndexingMsg', 916);

proc ark__slice__val_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3);
registerFunction('[slice]=val<int64,3>', ark__slice__val_int_3, 'IndexingMsg', 916);

proc ark__slice__val_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4);
registerFunction('[slice]=val<int64,4>', ark__slice__val_int_4, 'IndexingMsg', 916);

proc ark__slice__val_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('[slice]=val<uint64,1>', ark__slice__val_uint_1, 'IndexingMsg', 916);

proc ark__slice__val_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2);
registerFunction('[slice]=val<uint64,2>', ark__slice__val_uint_2, 'IndexingMsg', 916);

proc ark__slice__val_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3);
registerFunction('[slice]=val<uint64,3>', ark__slice__val_uint_3, 'IndexingMsg', 916);

proc ark__slice__val_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4);
registerFunction('[slice]=val<uint64,4>', ark__slice__val_uint_4, 'IndexingMsg', 916);

proc ark__slice__val_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('[slice]=val<uint8,1>', ark__slice__val_uint8_1, 'IndexingMsg', 916);

proc ark__slice__val_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2);
registerFunction('[slice]=val<uint8,2>', ark__slice__val_uint8_2, 'IndexingMsg', 916);

proc ark__slice__val_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3);
registerFunction('[slice]=val<uint8,3>', ark__slice__val_uint8_3, 'IndexingMsg', 916);

proc ark__slice__val_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4);
registerFunction('[slice]=val<uint8,4>', ark__slice__val_uint8_4, 'IndexingMsg', 916);

proc ark__slice__val_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('[slice]=val<float64,1>', ark__slice__val_real_1, 'IndexingMsg', 916);

proc ark__slice__val_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2);
registerFunction('[slice]=val<float64,2>', ark__slice__val_real_2, 'IndexingMsg', 916);

proc ark__slice__val_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3);
registerFunction('[slice]=val<float64,3>', ark__slice__val_real_3, 'IndexingMsg', 916);

proc ark__slice__val_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4);
registerFunction('[slice]=val<float64,4>', ark__slice__val_real_4, 'IndexingMsg', 916);

proc ark__slice__val_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('[slice]=val<bool,1>', ark__slice__val_bool_1, 'IndexingMsg', 916);

proc ark__slice__val_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2);
registerFunction('[slice]=val<bool,2>', ark__slice__val_bool_2, 'IndexingMsg', 916);

proc ark__slice__val_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3);
registerFunction('[slice]=val<bool,3>', ark__slice__val_bool_3, 'IndexingMsg', 916);

proc ark__slice__val_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4);
registerFunction('[slice]=val<bool,4>', ark__slice__val_bool_4, 'IndexingMsg', 916);

proc ark__slice__val_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('[slice]=val<bigint,1>', ark__slice__val_bigint_1, 'IndexingMsg', 916);

proc ark__slice__val_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2);
registerFunction('[slice]=val<bigint,2>', ark__slice__val_bigint_2, 'IndexingMsg', 916);

proc ark__slice__val_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3);
registerFunction('[slice]=val<bigint,3>', ark__slice__val_bigint_3, 'IndexingMsg', 916);

proc ark__slice__val_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4);
registerFunction('[slice]=val<bigint,4>', ark__slice__val_bigint_4, 'IndexingMsg', 916);

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

proc ark_arrayViewIntIndexAssign_int(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.arrayViewIntIndexAssign(cmd, msgArgs, st, array_dtype=int);
registerFunction('arrayViewIntIndexAssign<int64>', ark_arrayViewIntIndexAssign_int, 'IndexingMsg', 155);

proc ark_arrayViewIntIndexAssign_uint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.arrayViewIntIndexAssign(cmd, msgArgs, st, array_dtype=uint);
registerFunction('arrayViewIntIndexAssign<uint64>', ark_arrayViewIntIndexAssign_uint, 'IndexingMsg', 155);

proc ark_arrayViewIntIndexAssign_uint8(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.arrayViewIntIndexAssign(cmd, msgArgs, st, array_dtype=uint(8));
registerFunction('arrayViewIntIndexAssign<uint8>', ark_arrayViewIntIndexAssign_uint8, 'IndexingMsg', 155);

proc ark_arrayViewIntIndexAssign_real(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.arrayViewIntIndexAssign(cmd, msgArgs, st, array_dtype=real);
registerFunction('arrayViewIntIndexAssign<float64>', ark_arrayViewIntIndexAssign_real, 'IndexingMsg', 155);

proc ark_arrayViewIntIndexAssign_bool(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.arrayViewIntIndexAssign(cmd, msgArgs, st, array_dtype=bool);
registerFunction('arrayViewIntIndexAssign<bool>', ark_arrayViewIntIndexAssign_bool, 'IndexingMsg', 155);

proc ark_arrayViewIntIndexAssign_bigint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.arrayViewIntIndexAssign(cmd, msgArgs, st, array_dtype=bigint);
registerFunction('arrayViewIntIndexAssign<bigint>', ark_arrayViewIntIndexAssign_bigint, 'IndexingMsg', 155);

proc ark__pdarray__int_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=int, array_nd=1);
registerFunction('[pdarray]<int64,int64,1>', ark__pdarray__int_int_1, 'IndexingMsg', 246);

proc ark__pdarray__int_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=int, array_nd=2);
registerFunction('[pdarray]<int64,int64,2>', ark__pdarray__int_int_2, 'IndexingMsg', 246);

proc ark__pdarray__int_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=int, array_nd=3);
registerFunction('[pdarray]<int64,int64,3>', ark__pdarray__int_int_3, 'IndexingMsg', 246);

proc ark__pdarray__int_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=int, array_nd=4);
registerFunction('[pdarray]<int64,int64,4>', ark__pdarray__int_int_4, 'IndexingMsg', 246);

proc ark__pdarray__int_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=uint, array_nd=1);
registerFunction('[pdarray]<int64,uint64,1>', ark__pdarray__int_uint_1, 'IndexingMsg', 246);

proc ark__pdarray__int_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=uint, array_nd=2);
registerFunction('[pdarray]<int64,uint64,2>', ark__pdarray__int_uint_2, 'IndexingMsg', 246);

proc ark__pdarray__int_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=uint, array_nd=3);
registerFunction('[pdarray]<int64,uint64,3>', ark__pdarray__int_uint_3, 'IndexingMsg', 246);

proc ark__pdarray__int_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=uint, array_nd=4);
registerFunction('[pdarray]<int64,uint64,4>', ark__pdarray__int_uint_4, 'IndexingMsg', 246);

proc ark__pdarray__int_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=uint(8), array_nd=1);
registerFunction('[pdarray]<int64,uint8,1>', ark__pdarray__int_uint8_1, 'IndexingMsg', 246);

proc ark__pdarray__int_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=uint(8), array_nd=2);
registerFunction('[pdarray]<int64,uint8,2>', ark__pdarray__int_uint8_2, 'IndexingMsg', 246);

proc ark__pdarray__int_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=uint(8), array_nd=3);
registerFunction('[pdarray]<int64,uint8,3>', ark__pdarray__int_uint8_3, 'IndexingMsg', 246);

proc ark__pdarray__int_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=uint(8), array_nd=4);
registerFunction('[pdarray]<int64,uint8,4>', ark__pdarray__int_uint8_4, 'IndexingMsg', 246);

proc ark__pdarray__int_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=real, array_nd=1);
registerFunction('[pdarray]<int64,float64,1>', ark__pdarray__int_real_1, 'IndexingMsg', 246);

proc ark__pdarray__int_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=real, array_nd=2);
registerFunction('[pdarray]<int64,float64,2>', ark__pdarray__int_real_2, 'IndexingMsg', 246);

proc ark__pdarray__int_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=real, array_nd=3);
registerFunction('[pdarray]<int64,float64,3>', ark__pdarray__int_real_3, 'IndexingMsg', 246);

proc ark__pdarray__int_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=real, array_nd=4);
registerFunction('[pdarray]<int64,float64,4>', ark__pdarray__int_real_4, 'IndexingMsg', 246);

proc ark__pdarray__int_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=bool, array_nd=1);
registerFunction('[pdarray]<int64,bool,1>', ark__pdarray__int_bool_1, 'IndexingMsg', 246);

proc ark__pdarray__int_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=bool, array_nd=2);
registerFunction('[pdarray]<int64,bool,2>', ark__pdarray__int_bool_2, 'IndexingMsg', 246);

proc ark__pdarray__int_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=bool, array_nd=3);
registerFunction('[pdarray]<int64,bool,3>', ark__pdarray__int_bool_3, 'IndexingMsg', 246);

proc ark__pdarray__int_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=bool, array_nd=4);
registerFunction('[pdarray]<int64,bool,4>', ark__pdarray__int_bool_4, 'IndexingMsg', 246);

proc ark__pdarray__int_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=bigint, array_nd=1);
registerFunction('[pdarray]<int64,bigint,1>', ark__pdarray__int_bigint_1, 'IndexingMsg', 246);

proc ark__pdarray__int_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=bigint, array_nd=2);
registerFunction('[pdarray]<int64,bigint,2>', ark__pdarray__int_bigint_2, 'IndexingMsg', 246);

proc ark__pdarray__int_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=bigint, array_nd=3);
registerFunction('[pdarray]<int64,bigint,3>', ark__pdarray__int_bigint_3, 'IndexingMsg', 246);

proc ark__pdarray__int_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=bigint, array_nd=4);
registerFunction('[pdarray]<int64,bigint,4>', ark__pdarray__int_bigint_4, 'IndexingMsg', 246);

proc ark__pdarray__uint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=int, array_nd=1);
registerFunction('[pdarray]<uint64,int64,1>', ark__pdarray__uint_int_1, 'IndexingMsg', 246);

proc ark__pdarray__uint_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=int, array_nd=2);
registerFunction('[pdarray]<uint64,int64,2>', ark__pdarray__uint_int_2, 'IndexingMsg', 246);

proc ark__pdarray__uint_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=int, array_nd=3);
registerFunction('[pdarray]<uint64,int64,3>', ark__pdarray__uint_int_3, 'IndexingMsg', 246);

proc ark__pdarray__uint_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=int, array_nd=4);
registerFunction('[pdarray]<uint64,int64,4>', ark__pdarray__uint_int_4, 'IndexingMsg', 246);

proc ark__pdarray__uint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=uint, array_nd=1);
registerFunction('[pdarray]<uint64,uint64,1>', ark__pdarray__uint_uint_1, 'IndexingMsg', 246);

proc ark__pdarray__uint_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=uint, array_nd=2);
registerFunction('[pdarray]<uint64,uint64,2>', ark__pdarray__uint_uint_2, 'IndexingMsg', 246);

proc ark__pdarray__uint_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=uint, array_nd=3);
registerFunction('[pdarray]<uint64,uint64,3>', ark__pdarray__uint_uint_3, 'IndexingMsg', 246);

proc ark__pdarray__uint_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=uint, array_nd=4);
registerFunction('[pdarray]<uint64,uint64,4>', ark__pdarray__uint_uint_4, 'IndexingMsg', 246);

proc ark__pdarray__uint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=uint(8), array_nd=1);
registerFunction('[pdarray]<uint64,uint8,1>', ark__pdarray__uint_uint8_1, 'IndexingMsg', 246);

proc ark__pdarray__uint_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=uint(8), array_nd=2);
registerFunction('[pdarray]<uint64,uint8,2>', ark__pdarray__uint_uint8_2, 'IndexingMsg', 246);

proc ark__pdarray__uint_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=uint(8), array_nd=3);
registerFunction('[pdarray]<uint64,uint8,3>', ark__pdarray__uint_uint8_3, 'IndexingMsg', 246);

proc ark__pdarray__uint_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=uint(8), array_nd=4);
registerFunction('[pdarray]<uint64,uint8,4>', ark__pdarray__uint_uint8_4, 'IndexingMsg', 246);

proc ark__pdarray__uint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=real, array_nd=1);
registerFunction('[pdarray]<uint64,float64,1>', ark__pdarray__uint_real_1, 'IndexingMsg', 246);

proc ark__pdarray__uint_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=real, array_nd=2);
registerFunction('[pdarray]<uint64,float64,2>', ark__pdarray__uint_real_2, 'IndexingMsg', 246);

proc ark__pdarray__uint_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=real, array_nd=3);
registerFunction('[pdarray]<uint64,float64,3>', ark__pdarray__uint_real_3, 'IndexingMsg', 246);

proc ark__pdarray__uint_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=real, array_nd=4);
registerFunction('[pdarray]<uint64,float64,4>', ark__pdarray__uint_real_4, 'IndexingMsg', 246);

proc ark__pdarray__uint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=bool, array_nd=1);
registerFunction('[pdarray]<uint64,bool,1>', ark__pdarray__uint_bool_1, 'IndexingMsg', 246);

proc ark__pdarray__uint_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=bool, array_nd=2);
registerFunction('[pdarray]<uint64,bool,2>', ark__pdarray__uint_bool_2, 'IndexingMsg', 246);

proc ark__pdarray__uint_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=bool, array_nd=3);
registerFunction('[pdarray]<uint64,bool,3>', ark__pdarray__uint_bool_3, 'IndexingMsg', 246);

proc ark__pdarray__uint_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=bool, array_nd=4);
registerFunction('[pdarray]<uint64,bool,4>', ark__pdarray__uint_bool_4, 'IndexingMsg', 246);

proc ark__pdarray__uint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=bigint, array_nd=1);
registerFunction('[pdarray]<uint64,bigint,1>', ark__pdarray__uint_bigint_1, 'IndexingMsg', 246);

proc ark__pdarray__uint_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=bigint, array_nd=2);
registerFunction('[pdarray]<uint64,bigint,2>', ark__pdarray__uint_bigint_2, 'IndexingMsg', 246);

proc ark__pdarray__uint_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=bigint, array_nd=3);
registerFunction('[pdarray]<uint64,bigint,3>', ark__pdarray__uint_bigint_3, 'IndexingMsg', 246);

proc ark__pdarray__uint_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=bigint, array_nd=4);
registerFunction('[pdarray]<uint64,bigint,4>', ark__pdarray__uint_bigint_4, 'IndexingMsg', 246);

proc ark__pdarray__uint8_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=int, array_nd=1);
registerFunction('[pdarray]<uint8,int64,1>', ark__pdarray__uint8_int_1, 'IndexingMsg', 246);

proc ark__pdarray__uint8_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=int, array_nd=2);
registerFunction('[pdarray]<uint8,int64,2>', ark__pdarray__uint8_int_2, 'IndexingMsg', 246);

proc ark__pdarray__uint8_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=int, array_nd=3);
registerFunction('[pdarray]<uint8,int64,3>', ark__pdarray__uint8_int_3, 'IndexingMsg', 246);

proc ark__pdarray__uint8_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=int, array_nd=4);
registerFunction('[pdarray]<uint8,int64,4>', ark__pdarray__uint8_int_4, 'IndexingMsg', 246);

proc ark__pdarray__uint8_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=uint, array_nd=1);
registerFunction('[pdarray]<uint8,uint64,1>', ark__pdarray__uint8_uint_1, 'IndexingMsg', 246);

proc ark__pdarray__uint8_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=uint, array_nd=2);
registerFunction('[pdarray]<uint8,uint64,2>', ark__pdarray__uint8_uint_2, 'IndexingMsg', 246);

proc ark__pdarray__uint8_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=uint, array_nd=3);
registerFunction('[pdarray]<uint8,uint64,3>', ark__pdarray__uint8_uint_3, 'IndexingMsg', 246);

proc ark__pdarray__uint8_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=uint, array_nd=4);
registerFunction('[pdarray]<uint8,uint64,4>', ark__pdarray__uint8_uint_4, 'IndexingMsg', 246);

proc ark__pdarray__uint8_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=uint(8), array_nd=1);
registerFunction('[pdarray]<uint8,uint8,1>', ark__pdarray__uint8_uint8_1, 'IndexingMsg', 246);

proc ark__pdarray__uint8_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=uint(8), array_nd=2);
registerFunction('[pdarray]<uint8,uint8,2>', ark__pdarray__uint8_uint8_2, 'IndexingMsg', 246);

proc ark__pdarray__uint8_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=uint(8), array_nd=3);
registerFunction('[pdarray]<uint8,uint8,3>', ark__pdarray__uint8_uint8_3, 'IndexingMsg', 246);

proc ark__pdarray__uint8_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=uint(8), array_nd=4);
registerFunction('[pdarray]<uint8,uint8,4>', ark__pdarray__uint8_uint8_4, 'IndexingMsg', 246);

proc ark__pdarray__uint8_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=real, array_nd=1);
registerFunction('[pdarray]<uint8,float64,1>', ark__pdarray__uint8_real_1, 'IndexingMsg', 246);

proc ark__pdarray__uint8_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=real, array_nd=2);
registerFunction('[pdarray]<uint8,float64,2>', ark__pdarray__uint8_real_2, 'IndexingMsg', 246);

proc ark__pdarray__uint8_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=real, array_nd=3);
registerFunction('[pdarray]<uint8,float64,3>', ark__pdarray__uint8_real_3, 'IndexingMsg', 246);

proc ark__pdarray__uint8_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=real, array_nd=4);
registerFunction('[pdarray]<uint8,float64,4>', ark__pdarray__uint8_real_4, 'IndexingMsg', 246);

proc ark__pdarray__uint8_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=bool, array_nd=1);
registerFunction('[pdarray]<uint8,bool,1>', ark__pdarray__uint8_bool_1, 'IndexingMsg', 246);

proc ark__pdarray__uint8_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=bool, array_nd=2);
registerFunction('[pdarray]<uint8,bool,2>', ark__pdarray__uint8_bool_2, 'IndexingMsg', 246);

proc ark__pdarray__uint8_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=bool, array_nd=3);
registerFunction('[pdarray]<uint8,bool,3>', ark__pdarray__uint8_bool_3, 'IndexingMsg', 246);

proc ark__pdarray__uint8_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=bool, array_nd=4);
registerFunction('[pdarray]<uint8,bool,4>', ark__pdarray__uint8_bool_4, 'IndexingMsg', 246);

proc ark__pdarray__uint8_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=bigint, array_nd=1);
registerFunction('[pdarray]<uint8,bigint,1>', ark__pdarray__uint8_bigint_1, 'IndexingMsg', 246);

proc ark__pdarray__uint8_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=bigint, array_nd=2);
registerFunction('[pdarray]<uint8,bigint,2>', ark__pdarray__uint8_bigint_2, 'IndexingMsg', 246);

proc ark__pdarray__uint8_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=bigint, array_nd=3);
registerFunction('[pdarray]<uint8,bigint,3>', ark__pdarray__uint8_bigint_3, 'IndexingMsg', 246);

proc ark__pdarray__uint8_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=bigint, array_nd=4);
registerFunction('[pdarray]<uint8,bigint,4>', ark__pdarray__uint8_bigint_4, 'IndexingMsg', 246);

proc ark__pdarray__real_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=int, array_nd=1);
registerFunction('[pdarray]<float64,int64,1>', ark__pdarray__real_int_1, 'IndexingMsg', 246);

proc ark__pdarray__real_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=int, array_nd=2);
registerFunction('[pdarray]<float64,int64,2>', ark__pdarray__real_int_2, 'IndexingMsg', 246);

proc ark__pdarray__real_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=int, array_nd=3);
registerFunction('[pdarray]<float64,int64,3>', ark__pdarray__real_int_3, 'IndexingMsg', 246);

proc ark__pdarray__real_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=int, array_nd=4);
registerFunction('[pdarray]<float64,int64,4>', ark__pdarray__real_int_4, 'IndexingMsg', 246);

proc ark__pdarray__real_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=uint, array_nd=1);
registerFunction('[pdarray]<float64,uint64,1>', ark__pdarray__real_uint_1, 'IndexingMsg', 246);

proc ark__pdarray__real_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=uint, array_nd=2);
registerFunction('[pdarray]<float64,uint64,2>', ark__pdarray__real_uint_2, 'IndexingMsg', 246);

proc ark__pdarray__real_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=uint, array_nd=3);
registerFunction('[pdarray]<float64,uint64,3>', ark__pdarray__real_uint_3, 'IndexingMsg', 246);

proc ark__pdarray__real_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=uint, array_nd=4);
registerFunction('[pdarray]<float64,uint64,4>', ark__pdarray__real_uint_4, 'IndexingMsg', 246);

proc ark__pdarray__real_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=uint(8), array_nd=1);
registerFunction('[pdarray]<float64,uint8,1>', ark__pdarray__real_uint8_1, 'IndexingMsg', 246);

proc ark__pdarray__real_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=uint(8), array_nd=2);
registerFunction('[pdarray]<float64,uint8,2>', ark__pdarray__real_uint8_2, 'IndexingMsg', 246);

proc ark__pdarray__real_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=uint(8), array_nd=3);
registerFunction('[pdarray]<float64,uint8,3>', ark__pdarray__real_uint8_3, 'IndexingMsg', 246);

proc ark__pdarray__real_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=uint(8), array_nd=4);
registerFunction('[pdarray]<float64,uint8,4>', ark__pdarray__real_uint8_4, 'IndexingMsg', 246);

proc ark__pdarray__real_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=real, array_nd=1);
registerFunction('[pdarray]<float64,float64,1>', ark__pdarray__real_real_1, 'IndexingMsg', 246);

proc ark__pdarray__real_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=real, array_nd=2);
registerFunction('[pdarray]<float64,float64,2>', ark__pdarray__real_real_2, 'IndexingMsg', 246);

proc ark__pdarray__real_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=real, array_nd=3);
registerFunction('[pdarray]<float64,float64,3>', ark__pdarray__real_real_3, 'IndexingMsg', 246);

proc ark__pdarray__real_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=real, array_nd=4);
registerFunction('[pdarray]<float64,float64,4>', ark__pdarray__real_real_4, 'IndexingMsg', 246);

proc ark__pdarray__real_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=bool, array_nd=1);
registerFunction('[pdarray]<float64,bool,1>', ark__pdarray__real_bool_1, 'IndexingMsg', 246);

proc ark__pdarray__real_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=bool, array_nd=2);
registerFunction('[pdarray]<float64,bool,2>', ark__pdarray__real_bool_2, 'IndexingMsg', 246);

proc ark__pdarray__real_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=bool, array_nd=3);
registerFunction('[pdarray]<float64,bool,3>', ark__pdarray__real_bool_3, 'IndexingMsg', 246);

proc ark__pdarray__real_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=bool, array_nd=4);
registerFunction('[pdarray]<float64,bool,4>', ark__pdarray__real_bool_4, 'IndexingMsg', 246);

proc ark__pdarray__real_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=bigint, array_nd=1);
registerFunction('[pdarray]<float64,bigint,1>', ark__pdarray__real_bigint_1, 'IndexingMsg', 246);

proc ark__pdarray__real_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=bigint, array_nd=2);
registerFunction('[pdarray]<float64,bigint,2>', ark__pdarray__real_bigint_2, 'IndexingMsg', 246);

proc ark__pdarray__real_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=bigint, array_nd=3);
registerFunction('[pdarray]<float64,bigint,3>', ark__pdarray__real_bigint_3, 'IndexingMsg', 246);

proc ark__pdarray__real_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=bigint, array_nd=4);
registerFunction('[pdarray]<float64,bigint,4>', ark__pdarray__real_bigint_4, 'IndexingMsg', 246);

proc ark__pdarray__bool_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=int, array_nd=1);
registerFunction('[pdarray]<bool,int64,1>', ark__pdarray__bool_int_1, 'IndexingMsg', 246);

proc ark__pdarray__bool_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=int, array_nd=2);
registerFunction('[pdarray]<bool,int64,2>', ark__pdarray__bool_int_2, 'IndexingMsg', 246);

proc ark__pdarray__bool_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=int, array_nd=3);
registerFunction('[pdarray]<bool,int64,3>', ark__pdarray__bool_int_3, 'IndexingMsg', 246);

proc ark__pdarray__bool_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=int, array_nd=4);
registerFunction('[pdarray]<bool,int64,4>', ark__pdarray__bool_int_4, 'IndexingMsg', 246);

proc ark__pdarray__bool_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=uint, array_nd=1);
registerFunction('[pdarray]<bool,uint64,1>', ark__pdarray__bool_uint_1, 'IndexingMsg', 246);

proc ark__pdarray__bool_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=uint, array_nd=2);
registerFunction('[pdarray]<bool,uint64,2>', ark__pdarray__bool_uint_2, 'IndexingMsg', 246);

proc ark__pdarray__bool_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=uint, array_nd=3);
registerFunction('[pdarray]<bool,uint64,3>', ark__pdarray__bool_uint_3, 'IndexingMsg', 246);

proc ark__pdarray__bool_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=uint, array_nd=4);
registerFunction('[pdarray]<bool,uint64,4>', ark__pdarray__bool_uint_4, 'IndexingMsg', 246);

proc ark__pdarray__bool_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=uint(8), array_nd=1);
registerFunction('[pdarray]<bool,uint8,1>', ark__pdarray__bool_uint8_1, 'IndexingMsg', 246);

proc ark__pdarray__bool_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=uint(8), array_nd=2);
registerFunction('[pdarray]<bool,uint8,2>', ark__pdarray__bool_uint8_2, 'IndexingMsg', 246);

proc ark__pdarray__bool_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=uint(8), array_nd=3);
registerFunction('[pdarray]<bool,uint8,3>', ark__pdarray__bool_uint8_3, 'IndexingMsg', 246);

proc ark__pdarray__bool_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=uint(8), array_nd=4);
registerFunction('[pdarray]<bool,uint8,4>', ark__pdarray__bool_uint8_4, 'IndexingMsg', 246);

proc ark__pdarray__bool_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=real, array_nd=1);
registerFunction('[pdarray]<bool,float64,1>', ark__pdarray__bool_real_1, 'IndexingMsg', 246);

proc ark__pdarray__bool_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=real, array_nd=2);
registerFunction('[pdarray]<bool,float64,2>', ark__pdarray__bool_real_2, 'IndexingMsg', 246);

proc ark__pdarray__bool_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=real, array_nd=3);
registerFunction('[pdarray]<bool,float64,3>', ark__pdarray__bool_real_3, 'IndexingMsg', 246);

proc ark__pdarray__bool_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=real, array_nd=4);
registerFunction('[pdarray]<bool,float64,4>', ark__pdarray__bool_real_4, 'IndexingMsg', 246);

proc ark__pdarray__bool_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=bool, array_nd=1);
registerFunction('[pdarray]<bool,bool,1>', ark__pdarray__bool_bool_1, 'IndexingMsg', 246);

proc ark__pdarray__bool_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=bool, array_nd=2);
registerFunction('[pdarray]<bool,bool,2>', ark__pdarray__bool_bool_2, 'IndexingMsg', 246);

proc ark__pdarray__bool_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=bool, array_nd=3);
registerFunction('[pdarray]<bool,bool,3>', ark__pdarray__bool_bool_3, 'IndexingMsg', 246);

proc ark__pdarray__bool_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=bool, array_nd=4);
registerFunction('[pdarray]<bool,bool,4>', ark__pdarray__bool_bool_4, 'IndexingMsg', 246);

proc ark__pdarray__bool_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=bigint, array_nd=1);
registerFunction('[pdarray]<bool,bigint,1>', ark__pdarray__bool_bigint_1, 'IndexingMsg', 246);

proc ark__pdarray__bool_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=bigint, array_nd=2);
registerFunction('[pdarray]<bool,bigint,2>', ark__pdarray__bool_bigint_2, 'IndexingMsg', 246);

proc ark__pdarray__bool_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=bigint, array_nd=3);
registerFunction('[pdarray]<bool,bigint,3>', ark__pdarray__bool_bigint_3, 'IndexingMsg', 246);

proc ark__pdarray__bool_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=bigint, array_nd=4);
registerFunction('[pdarray]<bool,bigint,4>', ark__pdarray__bool_bigint_4, 'IndexingMsg', 246);

proc ark__pdarray__bigint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=int, array_nd=1);
registerFunction('[pdarray]<bigint,int64,1>', ark__pdarray__bigint_int_1, 'IndexingMsg', 246);

proc ark__pdarray__bigint_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=int, array_nd=2);
registerFunction('[pdarray]<bigint,int64,2>', ark__pdarray__bigint_int_2, 'IndexingMsg', 246);

proc ark__pdarray__bigint_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=int, array_nd=3);
registerFunction('[pdarray]<bigint,int64,3>', ark__pdarray__bigint_int_3, 'IndexingMsg', 246);

proc ark__pdarray__bigint_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=int, array_nd=4);
registerFunction('[pdarray]<bigint,int64,4>', ark__pdarray__bigint_int_4, 'IndexingMsg', 246);

proc ark__pdarray__bigint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=uint, array_nd=1);
registerFunction('[pdarray]<bigint,uint64,1>', ark__pdarray__bigint_uint_1, 'IndexingMsg', 246);

proc ark__pdarray__bigint_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=uint, array_nd=2);
registerFunction('[pdarray]<bigint,uint64,2>', ark__pdarray__bigint_uint_2, 'IndexingMsg', 246);

proc ark__pdarray__bigint_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=uint, array_nd=3);
registerFunction('[pdarray]<bigint,uint64,3>', ark__pdarray__bigint_uint_3, 'IndexingMsg', 246);

proc ark__pdarray__bigint_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=uint, array_nd=4);
registerFunction('[pdarray]<bigint,uint64,4>', ark__pdarray__bigint_uint_4, 'IndexingMsg', 246);

proc ark__pdarray__bigint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=uint(8), array_nd=1);
registerFunction('[pdarray]<bigint,uint8,1>', ark__pdarray__bigint_uint8_1, 'IndexingMsg', 246);

proc ark__pdarray__bigint_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=uint(8), array_nd=2);
registerFunction('[pdarray]<bigint,uint8,2>', ark__pdarray__bigint_uint8_2, 'IndexingMsg', 246);

proc ark__pdarray__bigint_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=uint(8), array_nd=3);
registerFunction('[pdarray]<bigint,uint8,3>', ark__pdarray__bigint_uint8_3, 'IndexingMsg', 246);

proc ark__pdarray__bigint_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=uint(8), array_nd=4);
registerFunction('[pdarray]<bigint,uint8,4>', ark__pdarray__bigint_uint8_4, 'IndexingMsg', 246);

proc ark__pdarray__bigint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=real, array_nd=1);
registerFunction('[pdarray]<bigint,float64,1>', ark__pdarray__bigint_real_1, 'IndexingMsg', 246);

proc ark__pdarray__bigint_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=real, array_nd=2);
registerFunction('[pdarray]<bigint,float64,2>', ark__pdarray__bigint_real_2, 'IndexingMsg', 246);

proc ark__pdarray__bigint_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=real, array_nd=3);
registerFunction('[pdarray]<bigint,float64,3>', ark__pdarray__bigint_real_3, 'IndexingMsg', 246);

proc ark__pdarray__bigint_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=real, array_nd=4);
registerFunction('[pdarray]<bigint,float64,4>', ark__pdarray__bigint_real_4, 'IndexingMsg', 246);

proc ark__pdarray__bigint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=bool, array_nd=1);
registerFunction('[pdarray]<bigint,bool,1>', ark__pdarray__bigint_bool_1, 'IndexingMsg', 246);

proc ark__pdarray__bigint_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=bool, array_nd=2);
registerFunction('[pdarray]<bigint,bool,2>', ark__pdarray__bigint_bool_2, 'IndexingMsg', 246);

proc ark__pdarray__bigint_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=bool, array_nd=3);
registerFunction('[pdarray]<bigint,bool,3>', ark__pdarray__bigint_bool_3, 'IndexingMsg', 246);

proc ark__pdarray__bigint_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=bool, array_nd=4);
registerFunction('[pdarray]<bigint,bool,4>', ark__pdarray__bigint_bool_4, 'IndexingMsg', 246);

proc ark__pdarray__bigint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=bigint, array_nd=1);
registerFunction('[pdarray]<bigint,bigint,1>', ark__pdarray__bigint_bigint_1, 'IndexingMsg', 246);

proc ark__pdarray__bigint_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=bigint, array_nd=2);
registerFunction('[pdarray]<bigint,bigint,2>', ark__pdarray__bigint_bigint_2, 'IndexingMsg', 246);

proc ark__pdarray__bigint_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=bigint, array_nd=3);
registerFunction('[pdarray]<bigint,bigint,3>', ark__pdarray__bigint_bigint_3, 'IndexingMsg', 246);

proc ark__pdarray__bigint_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=bigint, array_nd=4);
registerFunction('[pdarray]<bigint,bigint,4>', ark__pdarray__bigint_bigint_4, 'IndexingMsg', 246);

proc ark__slice__pdarray_int_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=int, array_nd=1);
registerFunction('[slice]=pdarray<int64,int64,1>', ark__slice__pdarray_int_int_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=int, array_nd=2);
registerFunction('[slice]=pdarray<int64,int64,2>', ark__slice__pdarray_int_int_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=int, array_nd=3);
registerFunction('[slice]=pdarray<int64,int64,3>', ark__slice__pdarray_int_int_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=int, array_nd=4);
registerFunction('[slice]=pdarray<int64,int64,4>', ark__slice__pdarray_int_int_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=uint, array_nd=1);
registerFunction('[slice]=pdarray<int64,uint64,1>', ark__slice__pdarray_int_uint_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=uint, array_nd=2);
registerFunction('[slice]=pdarray<int64,uint64,2>', ark__slice__pdarray_int_uint_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=uint, array_nd=3);
registerFunction('[slice]=pdarray<int64,uint64,3>', ark__slice__pdarray_int_uint_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=uint, array_nd=4);
registerFunction('[slice]=pdarray<int64,uint64,4>', ark__slice__pdarray_int_uint_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=uint(8), array_nd=1);
registerFunction('[slice]=pdarray<int64,uint8,1>', ark__slice__pdarray_int_uint8_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=uint(8), array_nd=2);
registerFunction('[slice]=pdarray<int64,uint8,2>', ark__slice__pdarray_int_uint8_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=uint(8), array_nd=3);
registerFunction('[slice]=pdarray<int64,uint8,3>', ark__slice__pdarray_int_uint8_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=uint(8), array_nd=4);
registerFunction('[slice]=pdarray<int64,uint8,4>', ark__slice__pdarray_int_uint8_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=real, array_nd=1);
registerFunction('[slice]=pdarray<int64,float64,1>', ark__slice__pdarray_int_real_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=real, array_nd=2);
registerFunction('[slice]=pdarray<int64,float64,2>', ark__slice__pdarray_int_real_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=real, array_nd=3);
registerFunction('[slice]=pdarray<int64,float64,3>', ark__slice__pdarray_int_real_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=real, array_nd=4);
registerFunction('[slice]=pdarray<int64,float64,4>', ark__slice__pdarray_int_real_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=bool, array_nd=1);
registerFunction('[slice]=pdarray<int64,bool,1>', ark__slice__pdarray_int_bool_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=bool, array_nd=2);
registerFunction('[slice]=pdarray<int64,bool,2>', ark__slice__pdarray_int_bool_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=bool, array_nd=3);
registerFunction('[slice]=pdarray<int64,bool,3>', ark__slice__pdarray_int_bool_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=bool, array_nd=4);
registerFunction('[slice]=pdarray<int64,bool,4>', ark__slice__pdarray_int_bool_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=bigint, array_nd=1);
registerFunction('[slice]=pdarray<int64,bigint,1>', ark__slice__pdarray_int_bigint_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=bigint, array_nd=2);
registerFunction('[slice]=pdarray<int64,bigint,2>', ark__slice__pdarray_int_bigint_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=bigint, array_nd=3);
registerFunction('[slice]=pdarray<int64,bigint,3>', ark__slice__pdarray_int_bigint_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_int_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=bigint, array_nd=4);
registerFunction('[slice]=pdarray<int64,bigint,4>', ark__slice__pdarray_int_bigint_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=int, array_nd=1);
registerFunction('[slice]=pdarray<uint64,int64,1>', ark__slice__pdarray_uint_int_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=int, array_nd=2);
registerFunction('[slice]=pdarray<uint64,int64,2>', ark__slice__pdarray_uint_int_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=int, array_nd=3);
registerFunction('[slice]=pdarray<uint64,int64,3>', ark__slice__pdarray_uint_int_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=int, array_nd=4);
registerFunction('[slice]=pdarray<uint64,int64,4>', ark__slice__pdarray_uint_int_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=uint, array_nd=1);
registerFunction('[slice]=pdarray<uint64,uint64,1>', ark__slice__pdarray_uint_uint_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=uint, array_nd=2);
registerFunction('[slice]=pdarray<uint64,uint64,2>', ark__slice__pdarray_uint_uint_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=uint, array_nd=3);
registerFunction('[slice]=pdarray<uint64,uint64,3>', ark__slice__pdarray_uint_uint_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=uint, array_nd=4);
registerFunction('[slice]=pdarray<uint64,uint64,4>', ark__slice__pdarray_uint_uint_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=uint(8), array_nd=1);
registerFunction('[slice]=pdarray<uint64,uint8,1>', ark__slice__pdarray_uint_uint8_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=uint(8), array_nd=2);
registerFunction('[slice]=pdarray<uint64,uint8,2>', ark__slice__pdarray_uint_uint8_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=uint(8), array_nd=3);
registerFunction('[slice]=pdarray<uint64,uint8,3>', ark__slice__pdarray_uint_uint8_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=uint(8), array_nd=4);
registerFunction('[slice]=pdarray<uint64,uint8,4>', ark__slice__pdarray_uint_uint8_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=real, array_nd=1);
registerFunction('[slice]=pdarray<uint64,float64,1>', ark__slice__pdarray_uint_real_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=real, array_nd=2);
registerFunction('[slice]=pdarray<uint64,float64,2>', ark__slice__pdarray_uint_real_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=real, array_nd=3);
registerFunction('[slice]=pdarray<uint64,float64,3>', ark__slice__pdarray_uint_real_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=real, array_nd=4);
registerFunction('[slice]=pdarray<uint64,float64,4>', ark__slice__pdarray_uint_real_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=bool, array_nd=1);
registerFunction('[slice]=pdarray<uint64,bool,1>', ark__slice__pdarray_uint_bool_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=bool, array_nd=2);
registerFunction('[slice]=pdarray<uint64,bool,2>', ark__slice__pdarray_uint_bool_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=bool, array_nd=3);
registerFunction('[slice]=pdarray<uint64,bool,3>', ark__slice__pdarray_uint_bool_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=bool, array_nd=4);
registerFunction('[slice]=pdarray<uint64,bool,4>', ark__slice__pdarray_uint_bool_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=bigint, array_nd=1);
registerFunction('[slice]=pdarray<uint64,bigint,1>', ark__slice__pdarray_uint_bigint_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=bigint, array_nd=2);
registerFunction('[slice]=pdarray<uint64,bigint,2>', ark__slice__pdarray_uint_bigint_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=bigint, array_nd=3);
registerFunction('[slice]=pdarray<uint64,bigint,3>', ark__slice__pdarray_uint_bigint_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=bigint, array_nd=4);
registerFunction('[slice]=pdarray<uint64,bigint,4>', ark__slice__pdarray_uint_bigint_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=int, array_nd=1);
registerFunction('[slice]=pdarray<uint8,int64,1>', ark__slice__pdarray_uint8_int_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=int, array_nd=2);
registerFunction('[slice]=pdarray<uint8,int64,2>', ark__slice__pdarray_uint8_int_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=int, array_nd=3);
registerFunction('[slice]=pdarray<uint8,int64,3>', ark__slice__pdarray_uint8_int_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=int, array_nd=4);
registerFunction('[slice]=pdarray<uint8,int64,4>', ark__slice__pdarray_uint8_int_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=uint, array_nd=1);
registerFunction('[slice]=pdarray<uint8,uint64,1>', ark__slice__pdarray_uint8_uint_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=uint, array_nd=2);
registerFunction('[slice]=pdarray<uint8,uint64,2>', ark__slice__pdarray_uint8_uint_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=uint, array_nd=3);
registerFunction('[slice]=pdarray<uint8,uint64,3>', ark__slice__pdarray_uint8_uint_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=uint, array_nd=4);
registerFunction('[slice]=pdarray<uint8,uint64,4>', ark__slice__pdarray_uint8_uint_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=uint(8), array_nd=1);
registerFunction('[slice]=pdarray<uint8,uint8,1>', ark__slice__pdarray_uint8_uint8_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=uint(8), array_nd=2);
registerFunction('[slice]=pdarray<uint8,uint8,2>', ark__slice__pdarray_uint8_uint8_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=uint(8), array_nd=3);
registerFunction('[slice]=pdarray<uint8,uint8,3>', ark__slice__pdarray_uint8_uint8_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=uint(8), array_nd=4);
registerFunction('[slice]=pdarray<uint8,uint8,4>', ark__slice__pdarray_uint8_uint8_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=real, array_nd=1);
registerFunction('[slice]=pdarray<uint8,float64,1>', ark__slice__pdarray_uint8_real_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=real, array_nd=2);
registerFunction('[slice]=pdarray<uint8,float64,2>', ark__slice__pdarray_uint8_real_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=real, array_nd=3);
registerFunction('[slice]=pdarray<uint8,float64,3>', ark__slice__pdarray_uint8_real_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=real, array_nd=4);
registerFunction('[slice]=pdarray<uint8,float64,4>', ark__slice__pdarray_uint8_real_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=bool, array_nd=1);
registerFunction('[slice]=pdarray<uint8,bool,1>', ark__slice__pdarray_uint8_bool_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=bool, array_nd=2);
registerFunction('[slice]=pdarray<uint8,bool,2>', ark__slice__pdarray_uint8_bool_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=bool, array_nd=3);
registerFunction('[slice]=pdarray<uint8,bool,3>', ark__slice__pdarray_uint8_bool_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=bool, array_nd=4);
registerFunction('[slice]=pdarray<uint8,bool,4>', ark__slice__pdarray_uint8_bool_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=bigint, array_nd=1);
registerFunction('[slice]=pdarray<uint8,bigint,1>', ark__slice__pdarray_uint8_bigint_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=bigint, array_nd=2);
registerFunction('[slice]=pdarray<uint8,bigint,2>', ark__slice__pdarray_uint8_bigint_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=bigint, array_nd=3);
registerFunction('[slice]=pdarray<uint8,bigint,3>', ark__slice__pdarray_uint8_bigint_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_uint8_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=bigint, array_nd=4);
registerFunction('[slice]=pdarray<uint8,bigint,4>', ark__slice__pdarray_uint8_bigint_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=int, array_nd=1);
registerFunction('[slice]=pdarray<float64,int64,1>', ark__slice__pdarray_real_int_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=int, array_nd=2);
registerFunction('[slice]=pdarray<float64,int64,2>', ark__slice__pdarray_real_int_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=int, array_nd=3);
registerFunction('[slice]=pdarray<float64,int64,3>', ark__slice__pdarray_real_int_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=int, array_nd=4);
registerFunction('[slice]=pdarray<float64,int64,4>', ark__slice__pdarray_real_int_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=uint, array_nd=1);
registerFunction('[slice]=pdarray<float64,uint64,1>', ark__slice__pdarray_real_uint_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=uint, array_nd=2);
registerFunction('[slice]=pdarray<float64,uint64,2>', ark__slice__pdarray_real_uint_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=uint, array_nd=3);
registerFunction('[slice]=pdarray<float64,uint64,3>', ark__slice__pdarray_real_uint_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=uint, array_nd=4);
registerFunction('[slice]=pdarray<float64,uint64,4>', ark__slice__pdarray_real_uint_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=uint(8), array_nd=1);
registerFunction('[slice]=pdarray<float64,uint8,1>', ark__slice__pdarray_real_uint8_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=uint(8), array_nd=2);
registerFunction('[slice]=pdarray<float64,uint8,2>', ark__slice__pdarray_real_uint8_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=uint(8), array_nd=3);
registerFunction('[slice]=pdarray<float64,uint8,3>', ark__slice__pdarray_real_uint8_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=uint(8), array_nd=4);
registerFunction('[slice]=pdarray<float64,uint8,4>', ark__slice__pdarray_real_uint8_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=real, array_nd=1);
registerFunction('[slice]=pdarray<float64,float64,1>', ark__slice__pdarray_real_real_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=real, array_nd=2);
registerFunction('[slice]=pdarray<float64,float64,2>', ark__slice__pdarray_real_real_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=real, array_nd=3);
registerFunction('[slice]=pdarray<float64,float64,3>', ark__slice__pdarray_real_real_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=real, array_nd=4);
registerFunction('[slice]=pdarray<float64,float64,4>', ark__slice__pdarray_real_real_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=bool, array_nd=1);
registerFunction('[slice]=pdarray<float64,bool,1>', ark__slice__pdarray_real_bool_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=bool, array_nd=2);
registerFunction('[slice]=pdarray<float64,bool,2>', ark__slice__pdarray_real_bool_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=bool, array_nd=3);
registerFunction('[slice]=pdarray<float64,bool,3>', ark__slice__pdarray_real_bool_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=bool, array_nd=4);
registerFunction('[slice]=pdarray<float64,bool,4>', ark__slice__pdarray_real_bool_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=bigint, array_nd=1);
registerFunction('[slice]=pdarray<float64,bigint,1>', ark__slice__pdarray_real_bigint_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=bigint, array_nd=2);
registerFunction('[slice]=pdarray<float64,bigint,2>', ark__slice__pdarray_real_bigint_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=bigint, array_nd=3);
registerFunction('[slice]=pdarray<float64,bigint,3>', ark__slice__pdarray_real_bigint_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_real_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=bigint, array_nd=4);
registerFunction('[slice]=pdarray<float64,bigint,4>', ark__slice__pdarray_real_bigint_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=int, array_nd=1);
registerFunction('[slice]=pdarray<bool,int64,1>', ark__slice__pdarray_bool_int_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=int, array_nd=2);
registerFunction('[slice]=pdarray<bool,int64,2>', ark__slice__pdarray_bool_int_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=int, array_nd=3);
registerFunction('[slice]=pdarray<bool,int64,3>', ark__slice__pdarray_bool_int_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=int, array_nd=4);
registerFunction('[slice]=pdarray<bool,int64,4>', ark__slice__pdarray_bool_int_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=uint, array_nd=1);
registerFunction('[slice]=pdarray<bool,uint64,1>', ark__slice__pdarray_bool_uint_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=uint, array_nd=2);
registerFunction('[slice]=pdarray<bool,uint64,2>', ark__slice__pdarray_bool_uint_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=uint, array_nd=3);
registerFunction('[slice]=pdarray<bool,uint64,3>', ark__slice__pdarray_bool_uint_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=uint, array_nd=4);
registerFunction('[slice]=pdarray<bool,uint64,4>', ark__slice__pdarray_bool_uint_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=uint(8), array_nd=1);
registerFunction('[slice]=pdarray<bool,uint8,1>', ark__slice__pdarray_bool_uint8_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=uint(8), array_nd=2);
registerFunction('[slice]=pdarray<bool,uint8,2>', ark__slice__pdarray_bool_uint8_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=uint(8), array_nd=3);
registerFunction('[slice]=pdarray<bool,uint8,3>', ark__slice__pdarray_bool_uint8_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=uint(8), array_nd=4);
registerFunction('[slice]=pdarray<bool,uint8,4>', ark__slice__pdarray_bool_uint8_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=real, array_nd=1);
registerFunction('[slice]=pdarray<bool,float64,1>', ark__slice__pdarray_bool_real_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=real, array_nd=2);
registerFunction('[slice]=pdarray<bool,float64,2>', ark__slice__pdarray_bool_real_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=real, array_nd=3);
registerFunction('[slice]=pdarray<bool,float64,3>', ark__slice__pdarray_bool_real_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=real, array_nd=4);
registerFunction('[slice]=pdarray<bool,float64,4>', ark__slice__pdarray_bool_real_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=bool, array_nd=1);
registerFunction('[slice]=pdarray<bool,bool,1>', ark__slice__pdarray_bool_bool_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=bool, array_nd=2);
registerFunction('[slice]=pdarray<bool,bool,2>', ark__slice__pdarray_bool_bool_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=bool, array_nd=3);
registerFunction('[slice]=pdarray<bool,bool,3>', ark__slice__pdarray_bool_bool_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=bool, array_nd=4);
registerFunction('[slice]=pdarray<bool,bool,4>', ark__slice__pdarray_bool_bool_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=bigint, array_nd=1);
registerFunction('[slice]=pdarray<bool,bigint,1>', ark__slice__pdarray_bool_bigint_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=bigint, array_nd=2);
registerFunction('[slice]=pdarray<bool,bigint,2>', ark__slice__pdarray_bool_bigint_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=bigint, array_nd=3);
registerFunction('[slice]=pdarray<bool,bigint,3>', ark__slice__pdarray_bool_bigint_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_bool_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=bigint, array_nd=4);
registerFunction('[slice]=pdarray<bool,bigint,4>', ark__slice__pdarray_bool_bigint_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=int, array_nd=1);
registerFunction('[slice]=pdarray<bigint,int64,1>', ark__slice__pdarray_bigint_int_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=int, array_nd=2);
registerFunction('[slice]=pdarray<bigint,int64,2>', ark__slice__pdarray_bigint_int_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=int, array_nd=3);
registerFunction('[slice]=pdarray<bigint,int64,3>', ark__slice__pdarray_bigint_int_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=int, array_nd=4);
registerFunction('[slice]=pdarray<bigint,int64,4>', ark__slice__pdarray_bigint_int_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=uint, array_nd=1);
registerFunction('[slice]=pdarray<bigint,uint64,1>', ark__slice__pdarray_bigint_uint_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=uint, array_nd=2);
registerFunction('[slice]=pdarray<bigint,uint64,2>', ark__slice__pdarray_bigint_uint_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=uint, array_nd=3);
registerFunction('[slice]=pdarray<bigint,uint64,3>', ark__slice__pdarray_bigint_uint_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=uint, array_nd=4);
registerFunction('[slice]=pdarray<bigint,uint64,4>', ark__slice__pdarray_bigint_uint_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=uint(8), array_nd=1);
registerFunction('[slice]=pdarray<bigint,uint8,1>', ark__slice__pdarray_bigint_uint8_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=uint(8), array_nd=2);
registerFunction('[slice]=pdarray<bigint,uint8,2>', ark__slice__pdarray_bigint_uint8_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=uint(8), array_nd=3);
registerFunction('[slice]=pdarray<bigint,uint8,3>', ark__slice__pdarray_bigint_uint8_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=uint(8), array_nd=4);
registerFunction('[slice]=pdarray<bigint,uint8,4>', ark__slice__pdarray_bigint_uint8_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=real, array_nd=1);
registerFunction('[slice]=pdarray<bigint,float64,1>', ark__slice__pdarray_bigint_real_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=real, array_nd=2);
registerFunction('[slice]=pdarray<bigint,float64,2>', ark__slice__pdarray_bigint_real_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=real, array_nd=3);
registerFunction('[slice]=pdarray<bigint,float64,3>', ark__slice__pdarray_bigint_real_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=real, array_nd=4);
registerFunction('[slice]=pdarray<bigint,float64,4>', ark__slice__pdarray_bigint_real_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=bool, array_nd=1);
registerFunction('[slice]=pdarray<bigint,bool,1>', ark__slice__pdarray_bigint_bool_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=bool, array_nd=2);
registerFunction('[slice]=pdarray<bigint,bool,2>', ark__slice__pdarray_bigint_bool_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=bool, array_nd=3);
registerFunction('[slice]=pdarray<bigint,bool,3>', ark__slice__pdarray_bigint_bool_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=bool, array_nd=4);
registerFunction('[slice]=pdarray<bigint,bool,4>', ark__slice__pdarray_bigint_bool_4, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=bigint, array_nd=1);
registerFunction('[slice]=pdarray<bigint,bigint,1>', ark__slice__pdarray_bigint_bigint_1, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=bigint, array_nd=2);
registerFunction('[slice]=pdarray<bigint,bigint,2>', ark__slice__pdarray_bigint_bigint_2, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=bigint, array_nd=3);
registerFunction('[slice]=pdarray<bigint,bigint,3>', ark__slice__pdarray_bigint_bigint_3, 'IndexingMsg', 937);

proc ark__slice__pdarray_bigint_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=bigint, array_nd=4);
registerFunction('[slice]=pdarray<bigint,bigint,4>', ark__slice__pdarray_bigint_bigint_4, 'IndexingMsg', 937);

proc ark_takeAlongAxis_int_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=int, array_nd=1);
registerFunction('takeAlongAxis<int64,int64,1>', ark_takeAlongAxis_int_int_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=int, array_nd=2);
registerFunction('takeAlongAxis<int64,int64,2>', ark_takeAlongAxis_int_int_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=int, array_nd=3);
registerFunction('takeAlongAxis<int64,int64,3>', ark_takeAlongAxis_int_int_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=int, array_nd=4);
registerFunction('takeAlongAxis<int64,int64,4>', ark_takeAlongAxis_int_int_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=uint, array_nd=1);
registerFunction('takeAlongAxis<int64,uint64,1>', ark_takeAlongAxis_int_uint_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=uint, array_nd=2);
registerFunction('takeAlongAxis<int64,uint64,2>', ark_takeAlongAxis_int_uint_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=uint, array_nd=3);
registerFunction('takeAlongAxis<int64,uint64,3>', ark_takeAlongAxis_int_uint_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=uint, array_nd=4);
registerFunction('takeAlongAxis<int64,uint64,4>', ark_takeAlongAxis_int_uint_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=uint(8), array_nd=1);
registerFunction('takeAlongAxis<int64,uint8,1>', ark_takeAlongAxis_int_uint8_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=uint(8), array_nd=2);
registerFunction('takeAlongAxis<int64,uint8,2>', ark_takeAlongAxis_int_uint8_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=uint(8), array_nd=3);
registerFunction('takeAlongAxis<int64,uint8,3>', ark_takeAlongAxis_int_uint8_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=uint(8), array_nd=4);
registerFunction('takeAlongAxis<int64,uint8,4>', ark_takeAlongAxis_int_uint8_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=real, array_nd=1);
registerFunction('takeAlongAxis<int64,float64,1>', ark_takeAlongAxis_int_real_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=real, array_nd=2);
registerFunction('takeAlongAxis<int64,float64,2>', ark_takeAlongAxis_int_real_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=real, array_nd=3);
registerFunction('takeAlongAxis<int64,float64,3>', ark_takeAlongAxis_int_real_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=real, array_nd=4);
registerFunction('takeAlongAxis<int64,float64,4>', ark_takeAlongAxis_int_real_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=bool, array_nd=1);
registerFunction('takeAlongAxis<int64,bool,1>', ark_takeAlongAxis_int_bool_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=bool, array_nd=2);
registerFunction('takeAlongAxis<int64,bool,2>', ark_takeAlongAxis_int_bool_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=bool, array_nd=3);
registerFunction('takeAlongAxis<int64,bool,3>', ark_takeAlongAxis_int_bool_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=bool, array_nd=4);
registerFunction('takeAlongAxis<int64,bool,4>', ark_takeAlongAxis_int_bool_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=bigint, array_nd=1);
registerFunction('takeAlongAxis<int64,bigint,1>', ark_takeAlongAxis_int_bigint_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=bigint, array_nd=2);
registerFunction('takeAlongAxis<int64,bigint,2>', ark_takeAlongAxis_int_bigint_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=bigint, array_nd=3);
registerFunction('takeAlongAxis<int64,bigint,3>', ark_takeAlongAxis_int_bigint_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_int_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=int, array_dtype_idx=bigint, array_nd=4);
registerFunction('takeAlongAxis<int64,bigint,4>', ark_takeAlongAxis_int_bigint_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=int, array_nd=1);
registerFunction('takeAlongAxis<uint64,int64,1>', ark_takeAlongAxis_uint_int_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=int, array_nd=2);
registerFunction('takeAlongAxis<uint64,int64,2>', ark_takeAlongAxis_uint_int_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=int, array_nd=3);
registerFunction('takeAlongAxis<uint64,int64,3>', ark_takeAlongAxis_uint_int_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=int, array_nd=4);
registerFunction('takeAlongAxis<uint64,int64,4>', ark_takeAlongAxis_uint_int_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=uint, array_nd=1);
registerFunction('takeAlongAxis<uint64,uint64,1>', ark_takeAlongAxis_uint_uint_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=uint, array_nd=2);
registerFunction('takeAlongAxis<uint64,uint64,2>', ark_takeAlongAxis_uint_uint_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=uint, array_nd=3);
registerFunction('takeAlongAxis<uint64,uint64,3>', ark_takeAlongAxis_uint_uint_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=uint, array_nd=4);
registerFunction('takeAlongAxis<uint64,uint64,4>', ark_takeAlongAxis_uint_uint_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=uint(8), array_nd=1);
registerFunction('takeAlongAxis<uint64,uint8,1>', ark_takeAlongAxis_uint_uint8_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=uint(8), array_nd=2);
registerFunction('takeAlongAxis<uint64,uint8,2>', ark_takeAlongAxis_uint_uint8_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=uint(8), array_nd=3);
registerFunction('takeAlongAxis<uint64,uint8,3>', ark_takeAlongAxis_uint_uint8_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=uint(8), array_nd=4);
registerFunction('takeAlongAxis<uint64,uint8,4>', ark_takeAlongAxis_uint_uint8_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=real, array_nd=1);
registerFunction('takeAlongAxis<uint64,float64,1>', ark_takeAlongAxis_uint_real_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=real, array_nd=2);
registerFunction('takeAlongAxis<uint64,float64,2>', ark_takeAlongAxis_uint_real_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=real, array_nd=3);
registerFunction('takeAlongAxis<uint64,float64,3>', ark_takeAlongAxis_uint_real_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=real, array_nd=4);
registerFunction('takeAlongAxis<uint64,float64,4>', ark_takeAlongAxis_uint_real_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=bool, array_nd=1);
registerFunction('takeAlongAxis<uint64,bool,1>', ark_takeAlongAxis_uint_bool_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=bool, array_nd=2);
registerFunction('takeAlongAxis<uint64,bool,2>', ark_takeAlongAxis_uint_bool_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=bool, array_nd=3);
registerFunction('takeAlongAxis<uint64,bool,3>', ark_takeAlongAxis_uint_bool_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=bool, array_nd=4);
registerFunction('takeAlongAxis<uint64,bool,4>', ark_takeAlongAxis_uint_bool_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=bigint, array_nd=1);
registerFunction('takeAlongAxis<uint64,bigint,1>', ark_takeAlongAxis_uint_bigint_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=bigint, array_nd=2);
registerFunction('takeAlongAxis<uint64,bigint,2>', ark_takeAlongAxis_uint_bigint_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=bigint, array_nd=3);
registerFunction('takeAlongAxis<uint64,bigint,3>', ark_takeAlongAxis_uint_bigint_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint, array_dtype_idx=bigint, array_nd=4);
registerFunction('takeAlongAxis<uint64,bigint,4>', ark_takeAlongAxis_uint_bigint_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=int, array_nd=1);
registerFunction('takeAlongAxis<uint8,int64,1>', ark_takeAlongAxis_uint8_int_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=int, array_nd=2);
registerFunction('takeAlongAxis<uint8,int64,2>', ark_takeAlongAxis_uint8_int_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=int, array_nd=3);
registerFunction('takeAlongAxis<uint8,int64,3>', ark_takeAlongAxis_uint8_int_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=int, array_nd=4);
registerFunction('takeAlongAxis<uint8,int64,4>', ark_takeAlongAxis_uint8_int_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=uint, array_nd=1);
registerFunction('takeAlongAxis<uint8,uint64,1>', ark_takeAlongAxis_uint8_uint_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=uint, array_nd=2);
registerFunction('takeAlongAxis<uint8,uint64,2>', ark_takeAlongAxis_uint8_uint_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=uint, array_nd=3);
registerFunction('takeAlongAxis<uint8,uint64,3>', ark_takeAlongAxis_uint8_uint_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=uint, array_nd=4);
registerFunction('takeAlongAxis<uint8,uint64,4>', ark_takeAlongAxis_uint8_uint_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=uint(8), array_nd=1);
registerFunction('takeAlongAxis<uint8,uint8,1>', ark_takeAlongAxis_uint8_uint8_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=uint(8), array_nd=2);
registerFunction('takeAlongAxis<uint8,uint8,2>', ark_takeAlongAxis_uint8_uint8_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=uint(8), array_nd=3);
registerFunction('takeAlongAxis<uint8,uint8,3>', ark_takeAlongAxis_uint8_uint8_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=uint(8), array_nd=4);
registerFunction('takeAlongAxis<uint8,uint8,4>', ark_takeAlongAxis_uint8_uint8_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=real, array_nd=1);
registerFunction('takeAlongAxis<uint8,float64,1>', ark_takeAlongAxis_uint8_real_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=real, array_nd=2);
registerFunction('takeAlongAxis<uint8,float64,2>', ark_takeAlongAxis_uint8_real_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=real, array_nd=3);
registerFunction('takeAlongAxis<uint8,float64,3>', ark_takeAlongAxis_uint8_real_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=real, array_nd=4);
registerFunction('takeAlongAxis<uint8,float64,4>', ark_takeAlongAxis_uint8_real_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=bool, array_nd=1);
registerFunction('takeAlongAxis<uint8,bool,1>', ark_takeAlongAxis_uint8_bool_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=bool, array_nd=2);
registerFunction('takeAlongAxis<uint8,bool,2>', ark_takeAlongAxis_uint8_bool_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=bool, array_nd=3);
registerFunction('takeAlongAxis<uint8,bool,3>', ark_takeAlongAxis_uint8_bool_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=bool, array_nd=4);
registerFunction('takeAlongAxis<uint8,bool,4>', ark_takeAlongAxis_uint8_bool_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=bigint, array_nd=1);
registerFunction('takeAlongAxis<uint8,bigint,1>', ark_takeAlongAxis_uint8_bigint_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=bigint, array_nd=2);
registerFunction('takeAlongAxis<uint8,bigint,2>', ark_takeAlongAxis_uint8_bigint_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=bigint, array_nd=3);
registerFunction('takeAlongAxis<uint8,bigint,3>', ark_takeAlongAxis_uint8_bigint_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_uint8_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=uint(8), array_dtype_idx=bigint, array_nd=4);
registerFunction('takeAlongAxis<uint8,bigint,4>', ark_takeAlongAxis_uint8_bigint_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=int, array_nd=1);
registerFunction('takeAlongAxis<float64,int64,1>', ark_takeAlongAxis_real_int_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=int, array_nd=2);
registerFunction('takeAlongAxis<float64,int64,2>', ark_takeAlongAxis_real_int_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=int, array_nd=3);
registerFunction('takeAlongAxis<float64,int64,3>', ark_takeAlongAxis_real_int_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=int, array_nd=4);
registerFunction('takeAlongAxis<float64,int64,4>', ark_takeAlongAxis_real_int_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=uint, array_nd=1);
registerFunction('takeAlongAxis<float64,uint64,1>', ark_takeAlongAxis_real_uint_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=uint, array_nd=2);
registerFunction('takeAlongAxis<float64,uint64,2>', ark_takeAlongAxis_real_uint_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=uint, array_nd=3);
registerFunction('takeAlongAxis<float64,uint64,3>', ark_takeAlongAxis_real_uint_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=uint, array_nd=4);
registerFunction('takeAlongAxis<float64,uint64,4>', ark_takeAlongAxis_real_uint_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=uint(8), array_nd=1);
registerFunction('takeAlongAxis<float64,uint8,1>', ark_takeAlongAxis_real_uint8_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=uint(8), array_nd=2);
registerFunction('takeAlongAxis<float64,uint8,2>', ark_takeAlongAxis_real_uint8_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=uint(8), array_nd=3);
registerFunction('takeAlongAxis<float64,uint8,3>', ark_takeAlongAxis_real_uint8_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=uint(8), array_nd=4);
registerFunction('takeAlongAxis<float64,uint8,4>', ark_takeAlongAxis_real_uint8_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=real, array_nd=1);
registerFunction('takeAlongAxis<float64,float64,1>', ark_takeAlongAxis_real_real_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=real, array_nd=2);
registerFunction('takeAlongAxis<float64,float64,2>', ark_takeAlongAxis_real_real_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=real, array_nd=3);
registerFunction('takeAlongAxis<float64,float64,3>', ark_takeAlongAxis_real_real_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=real, array_nd=4);
registerFunction('takeAlongAxis<float64,float64,4>', ark_takeAlongAxis_real_real_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=bool, array_nd=1);
registerFunction('takeAlongAxis<float64,bool,1>', ark_takeAlongAxis_real_bool_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=bool, array_nd=2);
registerFunction('takeAlongAxis<float64,bool,2>', ark_takeAlongAxis_real_bool_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=bool, array_nd=3);
registerFunction('takeAlongAxis<float64,bool,3>', ark_takeAlongAxis_real_bool_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=bool, array_nd=4);
registerFunction('takeAlongAxis<float64,bool,4>', ark_takeAlongAxis_real_bool_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=bigint, array_nd=1);
registerFunction('takeAlongAxis<float64,bigint,1>', ark_takeAlongAxis_real_bigint_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=bigint, array_nd=2);
registerFunction('takeAlongAxis<float64,bigint,2>', ark_takeAlongAxis_real_bigint_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=bigint, array_nd=3);
registerFunction('takeAlongAxis<float64,bigint,3>', ark_takeAlongAxis_real_bigint_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_real_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=real, array_dtype_idx=bigint, array_nd=4);
registerFunction('takeAlongAxis<float64,bigint,4>', ark_takeAlongAxis_real_bigint_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=int, array_nd=1);
registerFunction('takeAlongAxis<bool,int64,1>', ark_takeAlongAxis_bool_int_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=int, array_nd=2);
registerFunction('takeAlongAxis<bool,int64,2>', ark_takeAlongAxis_bool_int_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=int, array_nd=3);
registerFunction('takeAlongAxis<bool,int64,3>', ark_takeAlongAxis_bool_int_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=int, array_nd=4);
registerFunction('takeAlongAxis<bool,int64,4>', ark_takeAlongAxis_bool_int_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=uint, array_nd=1);
registerFunction('takeAlongAxis<bool,uint64,1>', ark_takeAlongAxis_bool_uint_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=uint, array_nd=2);
registerFunction('takeAlongAxis<bool,uint64,2>', ark_takeAlongAxis_bool_uint_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=uint, array_nd=3);
registerFunction('takeAlongAxis<bool,uint64,3>', ark_takeAlongAxis_bool_uint_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=uint, array_nd=4);
registerFunction('takeAlongAxis<bool,uint64,4>', ark_takeAlongAxis_bool_uint_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=uint(8), array_nd=1);
registerFunction('takeAlongAxis<bool,uint8,1>', ark_takeAlongAxis_bool_uint8_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=uint(8), array_nd=2);
registerFunction('takeAlongAxis<bool,uint8,2>', ark_takeAlongAxis_bool_uint8_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=uint(8), array_nd=3);
registerFunction('takeAlongAxis<bool,uint8,3>', ark_takeAlongAxis_bool_uint8_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=uint(8), array_nd=4);
registerFunction('takeAlongAxis<bool,uint8,4>', ark_takeAlongAxis_bool_uint8_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=real, array_nd=1);
registerFunction('takeAlongAxis<bool,float64,1>', ark_takeAlongAxis_bool_real_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=real, array_nd=2);
registerFunction('takeAlongAxis<bool,float64,2>', ark_takeAlongAxis_bool_real_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=real, array_nd=3);
registerFunction('takeAlongAxis<bool,float64,3>', ark_takeAlongAxis_bool_real_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=real, array_nd=4);
registerFunction('takeAlongAxis<bool,float64,4>', ark_takeAlongAxis_bool_real_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=bool, array_nd=1);
registerFunction('takeAlongAxis<bool,bool,1>', ark_takeAlongAxis_bool_bool_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=bool, array_nd=2);
registerFunction('takeAlongAxis<bool,bool,2>', ark_takeAlongAxis_bool_bool_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=bool, array_nd=3);
registerFunction('takeAlongAxis<bool,bool,3>', ark_takeAlongAxis_bool_bool_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=bool, array_nd=4);
registerFunction('takeAlongAxis<bool,bool,4>', ark_takeAlongAxis_bool_bool_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=bigint, array_nd=1);
registerFunction('takeAlongAxis<bool,bigint,1>', ark_takeAlongAxis_bool_bigint_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=bigint, array_nd=2);
registerFunction('takeAlongAxis<bool,bigint,2>', ark_takeAlongAxis_bool_bigint_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=bigint, array_nd=3);
registerFunction('takeAlongAxis<bool,bigint,3>', ark_takeAlongAxis_bool_bigint_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bool_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bool, array_dtype_idx=bigint, array_nd=4);
registerFunction('takeAlongAxis<bool,bigint,4>', ark_takeAlongAxis_bool_bigint_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=int, array_nd=1);
registerFunction('takeAlongAxis<bigint,int64,1>', ark_takeAlongAxis_bigint_int_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=int, array_nd=2);
registerFunction('takeAlongAxis<bigint,int64,2>', ark_takeAlongAxis_bigint_int_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=int, array_nd=3);
registerFunction('takeAlongAxis<bigint,int64,3>', ark_takeAlongAxis_bigint_int_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=int, array_nd=4);
registerFunction('takeAlongAxis<bigint,int64,4>', ark_takeAlongAxis_bigint_int_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=uint, array_nd=1);
registerFunction('takeAlongAxis<bigint,uint64,1>', ark_takeAlongAxis_bigint_uint_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=uint, array_nd=2);
registerFunction('takeAlongAxis<bigint,uint64,2>', ark_takeAlongAxis_bigint_uint_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=uint, array_nd=3);
registerFunction('takeAlongAxis<bigint,uint64,3>', ark_takeAlongAxis_bigint_uint_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=uint, array_nd=4);
registerFunction('takeAlongAxis<bigint,uint64,4>', ark_takeAlongAxis_bigint_uint_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=uint(8), array_nd=1);
registerFunction('takeAlongAxis<bigint,uint8,1>', ark_takeAlongAxis_bigint_uint8_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=uint(8), array_nd=2);
registerFunction('takeAlongAxis<bigint,uint8,2>', ark_takeAlongAxis_bigint_uint8_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=uint(8), array_nd=3);
registerFunction('takeAlongAxis<bigint,uint8,3>', ark_takeAlongAxis_bigint_uint8_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=uint(8), array_nd=4);
registerFunction('takeAlongAxis<bigint,uint8,4>', ark_takeAlongAxis_bigint_uint8_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=real, array_nd=1);
registerFunction('takeAlongAxis<bigint,float64,1>', ark_takeAlongAxis_bigint_real_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=real, array_nd=2);
registerFunction('takeAlongAxis<bigint,float64,2>', ark_takeAlongAxis_bigint_real_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=real, array_nd=3);
registerFunction('takeAlongAxis<bigint,float64,3>', ark_takeAlongAxis_bigint_real_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=real, array_nd=4);
registerFunction('takeAlongAxis<bigint,float64,4>', ark_takeAlongAxis_bigint_real_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=bool, array_nd=1);
registerFunction('takeAlongAxis<bigint,bool,1>', ark_takeAlongAxis_bigint_bool_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=bool, array_nd=2);
registerFunction('takeAlongAxis<bigint,bool,2>', ark_takeAlongAxis_bigint_bool_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=bool, array_nd=3);
registerFunction('takeAlongAxis<bigint,bool,3>', ark_takeAlongAxis_bigint_bool_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=bool, array_nd=4);
registerFunction('takeAlongAxis<bigint,bool,4>', ark_takeAlongAxis_bigint_bool_4, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=bigint, array_nd=1);
registerFunction('takeAlongAxis<bigint,bigint,1>', ark_takeAlongAxis_bigint_bigint_1, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=bigint, array_nd=2);
registerFunction('takeAlongAxis<bigint,bigint,2>', ark_takeAlongAxis_bigint_bigint_2, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=bigint, array_nd=3);
registerFunction('takeAlongAxis<bigint,bigint,3>', ark_takeAlongAxis_bigint_bigint_3, 'IndexingMsg', 991);

proc ark_takeAlongAxis_bigint_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.takeAlongAxis(cmd, msgArgs, st, array_dtype_x=bigint, array_dtype_idx=bigint, array_nd=4);
registerFunction('takeAlongAxis<bigint,bigint,4>', ark_takeAlongAxis_bigint_bigint_4, 'IndexingMsg', 991);

import LinalgMsg;

proc ark_eye_int(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.eye(cmd, msgArgs, st, array_dtype=int);
registerFunction('eye<int64>', ark_eye_int, 'LinalgMsg', 26);

proc ark_eye_uint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.eye(cmd, msgArgs, st, array_dtype=uint);
registerFunction('eye<uint64>', ark_eye_uint, 'LinalgMsg', 26);

proc ark_eye_uint8(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.eye(cmd, msgArgs, st, array_dtype=uint(8));
registerFunction('eye<uint8>', ark_eye_uint8, 'LinalgMsg', 26);

proc ark_eye_real(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.eye(cmd, msgArgs, st, array_dtype=real);
registerFunction('eye<float64>', ark_eye_real, 'LinalgMsg', 26);

proc ark_eye_bool(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.eye(cmd, msgArgs, st, array_dtype=bool);
registerFunction('eye<bool>', ark_eye_bool, 'LinalgMsg', 26);

proc ark_eye_bigint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.eye(cmd, msgArgs, st, array_dtype=bigint);
registerFunction('eye<bigint>', ark_eye_bigint, 'LinalgMsg', 26);

proc ark_tril_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('tril<int64,1>', ark_tril_int_1, 'LinalgMsg', 81);

proc ark_tril_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('tril<int64,2>', ark_tril_int_2, 'LinalgMsg', 81);

proc ark_tril_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('tril<int64,3>', ark_tril_int_3, 'LinalgMsg', 81);

proc ark_tril_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('tril<int64,4>', ark_tril_int_4, 'LinalgMsg', 81);

proc ark_tril_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('tril<uint64,1>', ark_tril_uint_1, 'LinalgMsg', 81);

proc ark_tril_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('tril<uint64,2>', ark_tril_uint_2, 'LinalgMsg', 81);

proc ark_tril_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('tril<uint64,3>', ark_tril_uint_3, 'LinalgMsg', 81);

proc ark_tril_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('tril<uint64,4>', ark_tril_uint_4, 'LinalgMsg', 81);

proc ark_tril_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('tril<uint8,1>', ark_tril_uint8_1, 'LinalgMsg', 81);

proc ark_tril_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('tril<uint8,2>', ark_tril_uint8_2, 'LinalgMsg', 81);

proc ark_tril_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('tril<uint8,3>', ark_tril_uint8_3, 'LinalgMsg', 81);

proc ark_tril_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('tril<uint8,4>', ark_tril_uint8_4, 'LinalgMsg', 81);

proc ark_tril_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('tril<float64,1>', ark_tril_real_1, 'LinalgMsg', 81);

proc ark_tril_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('tril<float64,2>', ark_tril_real_2, 'LinalgMsg', 81);

proc ark_tril_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('tril<float64,3>', ark_tril_real_3, 'LinalgMsg', 81);

proc ark_tril_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('tril<float64,4>', ark_tril_real_4, 'LinalgMsg', 81);

proc ark_tril_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('tril<bool,1>', ark_tril_bool_1, 'LinalgMsg', 81);

proc ark_tril_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('tril<bool,2>', ark_tril_bool_2, 'LinalgMsg', 81);

proc ark_tril_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('tril<bool,3>', ark_tril_bool_3, 'LinalgMsg', 81);

proc ark_tril_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('tril<bool,4>', ark_tril_bool_4, 'LinalgMsg', 81);

proc ark_tril_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('tril<bigint,1>', ark_tril_bigint_1, 'LinalgMsg', 81);

proc ark_tril_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('tril<bigint,2>', ark_tril_bigint_2, 'LinalgMsg', 81);

proc ark_tril_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('tril<bigint,3>', ark_tril_bigint_3, 'LinalgMsg', 81);

proc ark_tril_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.tril(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('tril<bigint,4>', ark_tril_bigint_4, 'LinalgMsg', 81);

proc ark_triu_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('triu<int64,1>', ark_triu_int_1, 'LinalgMsg', 94);

proc ark_triu_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('triu<int64,2>', ark_triu_int_2, 'LinalgMsg', 94);

proc ark_triu_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('triu<int64,3>', ark_triu_int_3, 'LinalgMsg', 94);

proc ark_triu_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('triu<int64,4>', ark_triu_int_4, 'LinalgMsg', 94);

proc ark_triu_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('triu<uint64,1>', ark_triu_uint_1, 'LinalgMsg', 94);

proc ark_triu_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('triu<uint64,2>', ark_triu_uint_2, 'LinalgMsg', 94);

proc ark_triu_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('triu<uint64,3>', ark_triu_uint_3, 'LinalgMsg', 94);

proc ark_triu_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('triu<uint64,4>', ark_triu_uint_4, 'LinalgMsg', 94);

proc ark_triu_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('triu<uint8,1>', ark_triu_uint8_1, 'LinalgMsg', 94);

proc ark_triu_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('triu<uint8,2>', ark_triu_uint8_2, 'LinalgMsg', 94);

proc ark_triu_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('triu<uint8,3>', ark_triu_uint8_3, 'LinalgMsg', 94);

proc ark_triu_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('triu<uint8,4>', ark_triu_uint8_4, 'LinalgMsg', 94);

proc ark_triu_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('triu<float64,1>', ark_triu_real_1, 'LinalgMsg', 94);

proc ark_triu_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('triu<float64,2>', ark_triu_real_2, 'LinalgMsg', 94);

proc ark_triu_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('triu<float64,3>', ark_triu_real_3, 'LinalgMsg', 94);

proc ark_triu_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('triu<float64,4>', ark_triu_real_4, 'LinalgMsg', 94);

proc ark_triu_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('triu<bool,1>', ark_triu_bool_1, 'LinalgMsg', 94);

proc ark_triu_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('triu<bool,2>', ark_triu_bool_2, 'LinalgMsg', 94);

proc ark_triu_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('triu<bool,3>', ark_triu_bool_3, 'LinalgMsg', 94);

proc ark_triu_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('triu<bool,4>', ark_triu_bool_4, 'LinalgMsg', 94);

proc ark_triu_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('triu<bigint,1>', ark_triu_bigint_1, 'LinalgMsg', 94);

proc ark_triu_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('triu<bigint,2>', ark_triu_bigint_2, 'LinalgMsg', 94);

proc ark_triu_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('triu<bigint,3>', ark_triu_bigint_3, 'LinalgMsg', 94);

proc ark_triu_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.triu(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('triu<bigint,4>', ark_triu_bigint_4, 'LinalgMsg', 94);

proc ark_matmul_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.matmul(cmd, msgArgs, st, array_nd=1);
registerFunction('matmul<1>', ark_matmul_1, 'LinalgMsg', 175);

proc ark_matmul_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.matmul(cmd, msgArgs, st, array_nd=2);
registerFunction('matmul<2>', ark_matmul_2, 'LinalgMsg', 175);

proc ark_matmul_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.matmul(cmd, msgArgs, st, array_nd=3);
registerFunction('matmul<3>', ark_matmul_3, 'LinalgMsg', 175);

proc ark_matmul_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.matmul(cmd, msgArgs, st, array_nd=4);
registerFunction('matmul<4>', ark_matmul_4, 'LinalgMsg', 175);

proc ark_transpose_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('transpose<int64,1>', ark_transpose_int_1, 'LinalgMsg', 330);

proc ark_transpose_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('transpose<int64,2>', ark_transpose_int_2, 'LinalgMsg', 330);

proc ark_transpose_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('transpose<int64,3>', ark_transpose_int_3, 'LinalgMsg', 330);

proc ark_transpose_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('transpose<int64,4>', ark_transpose_int_4, 'LinalgMsg', 330);

proc ark_transpose_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('transpose<uint64,1>', ark_transpose_uint_1, 'LinalgMsg', 330);

proc ark_transpose_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('transpose<uint64,2>', ark_transpose_uint_2, 'LinalgMsg', 330);

proc ark_transpose_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('transpose<uint64,3>', ark_transpose_uint_3, 'LinalgMsg', 330);

proc ark_transpose_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('transpose<uint64,4>', ark_transpose_uint_4, 'LinalgMsg', 330);

proc ark_transpose_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('transpose<uint8,1>', ark_transpose_uint8_1, 'LinalgMsg', 330);

proc ark_transpose_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('transpose<uint8,2>', ark_transpose_uint8_2, 'LinalgMsg', 330);

proc ark_transpose_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('transpose<uint8,3>', ark_transpose_uint8_3, 'LinalgMsg', 330);

proc ark_transpose_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('transpose<uint8,4>', ark_transpose_uint8_4, 'LinalgMsg', 330);

proc ark_transpose_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('transpose<float64,1>', ark_transpose_real_1, 'LinalgMsg', 330);

proc ark_transpose_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('transpose<float64,2>', ark_transpose_real_2, 'LinalgMsg', 330);

proc ark_transpose_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('transpose<float64,3>', ark_transpose_real_3, 'LinalgMsg', 330);

proc ark_transpose_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('transpose<float64,4>', ark_transpose_real_4, 'LinalgMsg', 330);

proc ark_transpose_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('transpose<bool,1>', ark_transpose_bool_1, 'LinalgMsg', 330);

proc ark_transpose_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('transpose<bool,2>', ark_transpose_bool_2, 'LinalgMsg', 330);

proc ark_transpose_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('transpose<bool,3>', ark_transpose_bool_3, 'LinalgMsg', 330);

proc ark_transpose_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('transpose<bool,4>', ark_transpose_bool_4, 'LinalgMsg', 330);

proc ark_transpose_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('transpose<bigint,1>', ark_transpose_bigint_1, 'LinalgMsg', 330);

proc ark_transpose_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('transpose<bigint,2>', ark_transpose_bigint_2, 'LinalgMsg', 330);

proc ark_transpose_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('transpose<bigint,3>', ark_transpose_bigint_3, 'LinalgMsg', 330);

proc ark_transpose_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.transpose(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('transpose<bigint,4>', ark_transpose_bigint_4, 'LinalgMsg', 330);

proc ark_vecdot_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.vecdot(cmd, msgArgs, st, array_nd=1);
registerFunction('vecdot<1>', ark_vecdot_1, 'LinalgMsg', 383);

proc ark_vecdot_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.vecdot(cmd, msgArgs, st, array_nd=2);
registerFunction('vecdot<2>', ark_vecdot_2, 'LinalgMsg', 383);

proc ark_vecdot_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.vecdot(cmd, msgArgs, st, array_nd=3);
registerFunction('vecdot<3>', ark_vecdot_3, 'LinalgMsg', 383);

proc ark_vecdot_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return LinalgMsg.vecdot(cmd, msgArgs, st, array_nd=4);
registerFunction('vecdot<4>', ark_vecdot_4, 'LinalgMsg', 383);

import ManipulationMsg;

proc ark_broadcast_int_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<int64,1,1>', ark_broadcast_int_1_1, 'ManipulationMsg', 61);

proc ark_broadcast_int_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=2);
registerFunction('broadcast<int64,1,2>', ark_broadcast_int_1_2, 'ManipulationMsg', 61);

proc ark_broadcast_int_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=3);
registerFunction('broadcast<int64,1,3>', ark_broadcast_int_1_3, 'ManipulationMsg', 61);

proc ark_broadcast_int_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=4);
registerFunction('broadcast<int64,1,4>', ark_broadcast_int_1_4, 'ManipulationMsg', 61);

proc ark_broadcast_int_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=1);
registerFunction('broadcast<int64,2,1>', ark_broadcast_int_2_1, 'ManipulationMsg', 61);

proc ark_broadcast_int_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=2);
registerFunction('broadcast<int64,2,2>', ark_broadcast_int_2_2, 'ManipulationMsg', 61);

proc ark_broadcast_int_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=3);
registerFunction('broadcast<int64,2,3>', ark_broadcast_int_2_3, 'ManipulationMsg', 61);

proc ark_broadcast_int_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=4);
registerFunction('broadcast<int64,2,4>', ark_broadcast_int_2_4, 'ManipulationMsg', 61);

proc ark_broadcast_int_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=1);
registerFunction('broadcast<int64,3,1>', ark_broadcast_int_3_1, 'ManipulationMsg', 61);

proc ark_broadcast_int_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=2);
registerFunction('broadcast<int64,3,2>', ark_broadcast_int_3_2, 'ManipulationMsg', 61);

proc ark_broadcast_int_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=3);
registerFunction('broadcast<int64,3,3>', ark_broadcast_int_3_3, 'ManipulationMsg', 61);

proc ark_broadcast_int_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=4);
registerFunction('broadcast<int64,3,4>', ark_broadcast_int_3_4, 'ManipulationMsg', 61);

proc ark_broadcast_int_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=4, array_nd_out=1);
registerFunction('broadcast<int64,4,1>', ark_broadcast_int_4_1, 'ManipulationMsg', 61);

proc ark_broadcast_int_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=4, array_nd_out=2);
registerFunction('broadcast<int64,4,2>', ark_broadcast_int_4_2, 'ManipulationMsg', 61);

proc ark_broadcast_int_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=4, array_nd_out=3);
registerFunction('broadcast<int64,4,3>', ark_broadcast_int_4_3, 'ManipulationMsg', 61);

proc ark_broadcast_int_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=4, array_nd_out=4);
registerFunction('broadcast<int64,4,4>', ark_broadcast_int_4_4, 'ManipulationMsg', 61);

proc ark_broadcast_uint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<uint64,1,1>', ark_broadcast_uint_1_1, 'ManipulationMsg', 61);

proc ark_broadcast_uint_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=2);
registerFunction('broadcast<uint64,1,2>', ark_broadcast_uint_1_2, 'ManipulationMsg', 61);

proc ark_broadcast_uint_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=3);
registerFunction('broadcast<uint64,1,3>', ark_broadcast_uint_1_3, 'ManipulationMsg', 61);

proc ark_broadcast_uint_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=4);
registerFunction('broadcast<uint64,1,4>', ark_broadcast_uint_1_4, 'ManipulationMsg', 61);

proc ark_broadcast_uint_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=2, array_nd_out=1);
registerFunction('broadcast<uint64,2,1>', ark_broadcast_uint_2_1, 'ManipulationMsg', 61);

proc ark_broadcast_uint_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=2, array_nd_out=2);
registerFunction('broadcast<uint64,2,2>', ark_broadcast_uint_2_2, 'ManipulationMsg', 61);

proc ark_broadcast_uint_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=2, array_nd_out=3);
registerFunction('broadcast<uint64,2,3>', ark_broadcast_uint_2_3, 'ManipulationMsg', 61);

proc ark_broadcast_uint_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=2, array_nd_out=4);
registerFunction('broadcast<uint64,2,4>', ark_broadcast_uint_2_4, 'ManipulationMsg', 61);

proc ark_broadcast_uint_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=3, array_nd_out=1);
registerFunction('broadcast<uint64,3,1>', ark_broadcast_uint_3_1, 'ManipulationMsg', 61);

proc ark_broadcast_uint_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=3, array_nd_out=2);
registerFunction('broadcast<uint64,3,2>', ark_broadcast_uint_3_2, 'ManipulationMsg', 61);

proc ark_broadcast_uint_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=3, array_nd_out=3);
registerFunction('broadcast<uint64,3,3>', ark_broadcast_uint_3_3, 'ManipulationMsg', 61);

proc ark_broadcast_uint_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=3, array_nd_out=4);
registerFunction('broadcast<uint64,3,4>', ark_broadcast_uint_3_4, 'ManipulationMsg', 61);

proc ark_broadcast_uint_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=4, array_nd_out=1);
registerFunction('broadcast<uint64,4,1>', ark_broadcast_uint_4_1, 'ManipulationMsg', 61);

proc ark_broadcast_uint_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=4, array_nd_out=2);
registerFunction('broadcast<uint64,4,2>', ark_broadcast_uint_4_2, 'ManipulationMsg', 61);

proc ark_broadcast_uint_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=4, array_nd_out=3);
registerFunction('broadcast<uint64,4,3>', ark_broadcast_uint_4_3, 'ManipulationMsg', 61);

proc ark_broadcast_uint_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=4, array_nd_out=4);
registerFunction('broadcast<uint64,4,4>', ark_broadcast_uint_4_4, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<uint8,1,1>', ark_broadcast_uint8_1_1, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=2);
registerFunction('broadcast<uint8,1,2>', ark_broadcast_uint8_1_2, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=3);
registerFunction('broadcast<uint8,1,3>', ark_broadcast_uint8_1_3, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=4);
registerFunction('broadcast<uint8,1,4>', ark_broadcast_uint8_1_4, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=2, array_nd_out=1);
registerFunction('broadcast<uint8,2,1>', ark_broadcast_uint8_2_1, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=2, array_nd_out=2);
registerFunction('broadcast<uint8,2,2>', ark_broadcast_uint8_2_2, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=2, array_nd_out=3);
registerFunction('broadcast<uint8,2,3>', ark_broadcast_uint8_2_3, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=2, array_nd_out=4);
registerFunction('broadcast<uint8,2,4>', ark_broadcast_uint8_2_4, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=3, array_nd_out=1);
registerFunction('broadcast<uint8,3,1>', ark_broadcast_uint8_3_1, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=3, array_nd_out=2);
registerFunction('broadcast<uint8,3,2>', ark_broadcast_uint8_3_2, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=3, array_nd_out=3);
registerFunction('broadcast<uint8,3,3>', ark_broadcast_uint8_3_3, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=3, array_nd_out=4);
registerFunction('broadcast<uint8,3,4>', ark_broadcast_uint8_3_4, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=4, array_nd_out=1);
registerFunction('broadcast<uint8,4,1>', ark_broadcast_uint8_4_1, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=4, array_nd_out=2);
registerFunction('broadcast<uint8,4,2>', ark_broadcast_uint8_4_2, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=4, array_nd_out=3);
registerFunction('broadcast<uint8,4,3>', ark_broadcast_uint8_4_3, 'ManipulationMsg', 61);

proc ark_broadcast_uint8_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=4, array_nd_out=4);
registerFunction('broadcast<uint8,4,4>', ark_broadcast_uint8_4_4, 'ManipulationMsg', 61);

proc ark_broadcast_real_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<float64,1,1>', ark_broadcast_real_1_1, 'ManipulationMsg', 61);

proc ark_broadcast_real_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=2);
registerFunction('broadcast<float64,1,2>', ark_broadcast_real_1_2, 'ManipulationMsg', 61);

proc ark_broadcast_real_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=3);
registerFunction('broadcast<float64,1,3>', ark_broadcast_real_1_3, 'ManipulationMsg', 61);

proc ark_broadcast_real_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=4);
registerFunction('broadcast<float64,1,4>', ark_broadcast_real_1_4, 'ManipulationMsg', 61);

proc ark_broadcast_real_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=1);
registerFunction('broadcast<float64,2,1>', ark_broadcast_real_2_1, 'ManipulationMsg', 61);

proc ark_broadcast_real_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=2);
registerFunction('broadcast<float64,2,2>', ark_broadcast_real_2_2, 'ManipulationMsg', 61);

proc ark_broadcast_real_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=3);
registerFunction('broadcast<float64,2,3>', ark_broadcast_real_2_3, 'ManipulationMsg', 61);

proc ark_broadcast_real_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=4);
registerFunction('broadcast<float64,2,4>', ark_broadcast_real_2_4, 'ManipulationMsg', 61);

proc ark_broadcast_real_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=1);
registerFunction('broadcast<float64,3,1>', ark_broadcast_real_3_1, 'ManipulationMsg', 61);

proc ark_broadcast_real_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=2);
registerFunction('broadcast<float64,3,2>', ark_broadcast_real_3_2, 'ManipulationMsg', 61);

proc ark_broadcast_real_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=3);
registerFunction('broadcast<float64,3,3>', ark_broadcast_real_3_3, 'ManipulationMsg', 61);

proc ark_broadcast_real_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=4);
registerFunction('broadcast<float64,3,4>', ark_broadcast_real_3_4, 'ManipulationMsg', 61);

proc ark_broadcast_real_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=4, array_nd_out=1);
registerFunction('broadcast<float64,4,1>', ark_broadcast_real_4_1, 'ManipulationMsg', 61);

proc ark_broadcast_real_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=4, array_nd_out=2);
registerFunction('broadcast<float64,4,2>', ark_broadcast_real_4_2, 'ManipulationMsg', 61);

proc ark_broadcast_real_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=4, array_nd_out=3);
registerFunction('broadcast<float64,4,3>', ark_broadcast_real_4_3, 'ManipulationMsg', 61);

proc ark_broadcast_real_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=4, array_nd_out=4);
registerFunction('broadcast<float64,4,4>', ark_broadcast_real_4_4, 'ManipulationMsg', 61);

proc ark_broadcast_bool_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<bool,1,1>', ark_broadcast_bool_1_1, 'ManipulationMsg', 61);

proc ark_broadcast_bool_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=2);
registerFunction('broadcast<bool,1,2>', ark_broadcast_bool_1_2, 'ManipulationMsg', 61);

proc ark_broadcast_bool_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=3);
registerFunction('broadcast<bool,1,3>', ark_broadcast_bool_1_3, 'ManipulationMsg', 61);

proc ark_broadcast_bool_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=4);
registerFunction('broadcast<bool,1,4>', ark_broadcast_bool_1_4, 'ManipulationMsg', 61);

proc ark_broadcast_bool_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=2, array_nd_out=1);
registerFunction('broadcast<bool,2,1>', ark_broadcast_bool_2_1, 'ManipulationMsg', 61);

proc ark_broadcast_bool_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=2, array_nd_out=2);
registerFunction('broadcast<bool,2,2>', ark_broadcast_bool_2_2, 'ManipulationMsg', 61);

proc ark_broadcast_bool_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=2, array_nd_out=3);
registerFunction('broadcast<bool,2,3>', ark_broadcast_bool_2_3, 'ManipulationMsg', 61);

proc ark_broadcast_bool_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=2, array_nd_out=4);
registerFunction('broadcast<bool,2,4>', ark_broadcast_bool_2_4, 'ManipulationMsg', 61);

proc ark_broadcast_bool_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=3, array_nd_out=1);
registerFunction('broadcast<bool,3,1>', ark_broadcast_bool_3_1, 'ManipulationMsg', 61);

proc ark_broadcast_bool_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=3, array_nd_out=2);
registerFunction('broadcast<bool,3,2>', ark_broadcast_bool_3_2, 'ManipulationMsg', 61);

proc ark_broadcast_bool_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=3, array_nd_out=3);
registerFunction('broadcast<bool,3,3>', ark_broadcast_bool_3_3, 'ManipulationMsg', 61);

proc ark_broadcast_bool_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=3, array_nd_out=4);
registerFunction('broadcast<bool,3,4>', ark_broadcast_bool_3_4, 'ManipulationMsg', 61);

proc ark_broadcast_bool_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=4, array_nd_out=1);
registerFunction('broadcast<bool,4,1>', ark_broadcast_bool_4_1, 'ManipulationMsg', 61);

proc ark_broadcast_bool_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=4, array_nd_out=2);
registerFunction('broadcast<bool,4,2>', ark_broadcast_bool_4_2, 'ManipulationMsg', 61);

proc ark_broadcast_bool_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=4, array_nd_out=3);
registerFunction('broadcast<bool,4,3>', ark_broadcast_bool_4_3, 'ManipulationMsg', 61);

proc ark_broadcast_bool_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=4, array_nd_out=4);
registerFunction('broadcast<bool,4,4>', ark_broadcast_bool_4_4, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<bigint,1,1>', ark_broadcast_bigint_1_1, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=2);
registerFunction('broadcast<bigint,1,2>', ark_broadcast_bigint_1_2, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=3);
registerFunction('broadcast<bigint,1,3>', ark_broadcast_bigint_1_3, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=4);
registerFunction('broadcast<bigint,1,4>', ark_broadcast_bigint_1_4, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=2, array_nd_out=1);
registerFunction('broadcast<bigint,2,1>', ark_broadcast_bigint_2_1, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=2, array_nd_out=2);
registerFunction('broadcast<bigint,2,2>', ark_broadcast_bigint_2_2, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=2, array_nd_out=3);
registerFunction('broadcast<bigint,2,3>', ark_broadcast_bigint_2_3, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=2, array_nd_out=4);
registerFunction('broadcast<bigint,2,4>', ark_broadcast_bigint_2_4, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=3, array_nd_out=1);
registerFunction('broadcast<bigint,3,1>', ark_broadcast_bigint_3_1, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=3, array_nd_out=2);
registerFunction('broadcast<bigint,3,2>', ark_broadcast_bigint_3_2, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=3, array_nd_out=3);
registerFunction('broadcast<bigint,3,3>', ark_broadcast_bigint_3_3, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=3, array_nd_out=4);
registerFunction('broadcast<bigint,3,4>', ark_broadcast_bigint_3_4, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=4, array_nd_out=1);
registerFunction('broadcast<bigint,4,1>', ark_broadcast_bigint_4_1, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=4, array_nd_out=2);
registerFunction('broadcast<bigint,4,2>', ark_broadcast_bigint_4_2, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=4, array_nd_out=3);
registerFunction('broadcast<bigint,4,3>', ark_broadcast_bigint_4_3, 'ManipulationMsg', 61);

proc ark_broadcast_bigint_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=4, array_nd_out=4);
registerFunction('broadcast<bigint,4,4>', ark_broadcast_bigint_4_4, 'ManipulationMsg', 61);

proc ark_concat_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('concat<int64,1>', ark_concat_int_1, 'ManipulationMsg', 158);

proc ark_concat_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('concat<int64,2>', ark_concat_int_2, 'ManipulationMsg', 158);

proc ark_concat_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('concat<int64,3>', ark_concat_int_3, 'ManipulationMsg', 158);

proc ark_concat_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('concat<int64,4>', ark_concat_int_4, 'ManipulationMsg', 158);

proc ark_concat_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('concat<uint64,1>', ark_concat_uint_1, 'ManipulationMsg', 158);

proc ark_concat_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('concat<uint64,2>', ark_concat_uint_2, 'ManipulationMsg', 158);

proc ark_concat_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('concat<uint64,3>', ark_concat_uint_3, 'ManipulationMsg', 158);

proc ark_concat_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('concat<uint64,4>', ark_concat_uint_4, 'ManipulationMsg', 158);

proc ark_concat_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('concat<uint8,1>', ark_concat_uint8_1, 'ManipulationMsg', 158);

proc ark_concat_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('concat<uint8,2>', ark_concat_uint8_2, 'ManipulationMsg', 158);

proc ark_concat_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('concat<uint8,3>', ark_concat_uint8_3, 'ManipulationMsg', 158);

proc ark_concat_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('concat<uint8,4>', ark_concat_uint8_4, 'ManipulationMsg', 158);

proc ark_concat_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('concat<float64,1>', ark_concat_real_1, 'ManipulationMsg', 158);

proc ark_concat_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('concat<float64,2>', ark_concat_real_2, 'ManipulationMsg', 158);

proc ark_concat_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('concat<float64,3>', ark_concat_real_3, 'ManipulationMsg', 158);

proc ark_concat_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('concat<float64,4>', ark_concat_real_4, 'ManipulationMsg', 158);

proc ark_concat_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('concat<bool,1>', ark_concat_bool_1, 'ManipulationMsg', 158);

proc ark_concat_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('concat<bool,2>', ark_concat_bool_2, 'ManipulationMsg', 158);

proc ark_concat_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('concat<bool,3>', ark_concat_bool_3, 'ManipulationMsg', 158);

proc ark_concat_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('concat<bool,4>', ark_concat_bool_4, 'ManipulationMsg', 158);

proc ark_concat_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('concat<bigint,1>', ark_concat_bigint_1, 'ManipulationMsg', 158);

proc ark_concat_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('concat<bigint,2>', ark_concat_bigint_2, 'ManipulationMsg', 158);

proc ark_concat_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('concat<bigint,3>', ark_concat_bigint_3, 'ManipulationMsg', 158);

proc ark_concat_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('concat<bigint,4>', ark_concat_bigint_4, 'ManipulationMsg', 158);

proc ark_concatFlat_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('concatFlat<int64,1>', ark_concatFlat_int_1, 'ManipulationMsg', 214);

proc ark_concatFlat_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('concatFlat<int64,2>', ark_concatFlat_int_2, 'ManipulationMsg', 214);

proc ark_concatFlat_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('concatFlat<int64,3>', ark_concatFlat_int_3, 'ManipulationMsg', 214);

proc ark_concatFlat_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('concatFlat<int64,4>', ark_concatFlat_int_4, 'ManipulationMsg', 214);

proc ark_concatFlat_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('concatFlat<uint64,1>', ark_concatFlat_uint_1, 'ManipulationMsg', 214);

proc ark_concatFlat_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('concatFlat<uint64,2>', ark_concatFlat_uint_2, 'ManipulationMsg', 214);

proc ark_concatFlat_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('concatFlat<uint64,3>', ark_concatFlat_uint_3, 'ManipulationMsg', 214);

proc ark_concatFlat_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('concatFlat<uint64,4>', ark_concatFlat_uint_4, 'ManipulationMsg', 214);

proc ark_concatFlat_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('concatFlat<uint8,1>', ark_concatFlat_uint8_1, 'ManipulationMsg', 214);

proc ark_concatFlat_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('concatFlat<uint8,2>', ark_concatFlat_uint8_2, 'ManipulationMsg', 214);

proc ark_concatFlat_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('concatFlat<uint8,3>', ark_concatFlat_uint8_3, 'ManipulationMsg', 214);

proc ark_concatFlat_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('concatFlat<uint8,4>', ark_concatFlat_uint8_4, 'ManipulationMsg', 214);

proc ark_concatFlat_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('concatFlat<float64,1>', ark_concatFlat_real_1, 'ManipulationMsg', 214);

proc ark_concatFlat_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('concatFlat<float64,2>', ark_concatFlat_real_2, 'ManipulationMsg', 214);

proc ark_concatFlat_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('concatFlat<float64,3>', ark_concatFlat_real_3, 'ManipulationMsg', 214);

proc ark_concatFlat_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('concatFlat<float64,4>', ark_concatFlat_real_4, 'ManipulationMsg', 214);

proc ark_concatFlat_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('concatFlat<bool,1>', ark_concatFlat_bool_1, 'ManipulationMsg', 214);

proc ark_concatFlat_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('concatFlat<bool,2>', ark_concatFlat_bool_2, 'ManipulationMsg', 214);

proc ark_concatFlat_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('concatFlat<bool,3>', ark_concatFlat_bool_3, 'ManipulationMsg', 214);

proc ark_concatFlat_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('concatFlat<bool,4>', ark_concatFlat_bool_4, 'ManipulationMsg', 214);

proc ark_concatFlat_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('concatFlat<bigint,1>', ark_concatFlat_bigint_1, 'ManipulationMsg', 214);

proc ark_concatFlat_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('concatFlat<bigint,2>', ark_concatFlat_bigint_2, 'ManipulationMsg', 214);

proc ark_concatFlat_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('concatFlat<bigint,3>', ark_concatFlat_bigint_3, 'ManipulationMsg', 214);

proc ark_concatFlat_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('concatFlat<bigint,4>', ark_concatFlat_bigint_4, 'ManipulationMsg', 214);

proc ark_expandDims_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('expandDims<int64,1>', ark_expandDims_int_1, 'ManipulationMsg', 238);

proc ark_expandDims_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('expandDims<int64,2>', ark_expandDims_int_2, 'ManipulationMsg', 238);

proc ark_expandDims_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('expandDims<int64,3>', ark_expandDims_int_3, 'ManipulationMsg', 238);

proc ark_expandDims_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('expandDims<int64,4>', ark_expandDims_int_4, 'ManipulationMsg', 238);

proc ark_expandDims_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('expandDims<uint64,1>', ark_expandDims_uint_1, 'ManipulationMsg', 238);

proc ark_expandDims_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('expandDims<uint64,2>', ark_expandDims_uint_2, 'ManipulationMsg', 238);

proc ark_expandDims_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('expandDims<uint64,3>', ark_expandDims_uint_3, 'ManipulationMsg', 238);

proc ark_expandDims_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('expandDims<uint64,4>', ark_expandDims_uint_4, 'ManipulationMsg', 238);

proc ark_expandDims_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('expandDims<uint8,1>', ark_expandDims_uint8_1, 'ManipulationMsg', 238);

proc ark_expandDims_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('expandDims<uint8,2>', ark_expandDims_uint8_2, 'ManipulationMsg', 238);

proc ark_expandDims_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('expandDims<uint8,3>', ark_expandDims_uint8_3, 'ManipulationMsg', 238);

proc ark_expandDims_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('expandDims<uint8,4>', ark_expandDims_uint8_4, 'ManipulationMsg', 238);

proc ark_expandDims_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('expandDims<float64,1>', ark_expandDims_real_1, 'ManipulationMsg', 238);

proc ark_expandDims_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('expandDims<float64,2>', ark_expandDims_real_2, 'ManipulationMsg', 238);

proc ark_expandDims_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('expandDims<float64,3>', ark_expandDims_real_3, 'ManipulationMsg', 238);

proc ark_expandDims_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('expandDims<float64,4>', ark_expandDims_real_4, 'ManipulationMsg', 238);

proc ark_expandDims_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('expandDims<bool,1>', ark_expandDims_bool_1, 'ManipulationMsg', 238);

proc ark_expandDims_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('expandDims<bool,2>', ark_expandDims_bool_2, 'ManipulationMsg', 238);

proc ark_expandDims_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('expandDims<bool,3>', ark_expandDims_bool_3, 'ManipulationMsg', 238);

proc ark_expandDims_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('expandDims<bool,4>', ark_expandDims_bool_4, 'ManipulationMsg', 238);

proc ark_expandDims_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('expandDims<bigint,1>', ark_expandDims_bigint_1, 'ManipulationMsg', 238);

proc ark_expandDims_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('expandDims<bigint,2>', ark_expandDims_bigint_2, 'ManipulationMsg', 238);

proc ark_expandDims_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('expandDims<bigint,3>', ark_expandDims_bigint_3, 'ManipulationMsg', 238);

proc ark_expandDims_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('expandDims<bigint,4>', ark_expandDims_bigint_4, 'ManipulationMsg', 238);

proc ark_flip_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('flip<int64,1>', ark_flip_int_1, 'ManipulationMsg', 290);

proc ark_flip_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('flip<int64,2>', ark_flip_int_2, 'ManipulationMsg', 290);

proc ark_flip_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('flip<int64,3>', ark_flip_int_3, 'ManipulationMsg', 290);

proc ark_flip_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('flip<int64,4>', ark_flip_int_4, 'ManipulationMsg', 290);

proc ark_flip_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('flip<uint64,1>', ark_flip_uint_1, 'ManipulationMsg', 290);

proc ark_flip_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('flip<uint64,2>', ark_flip_uint_2, 'ManipulationMsg', 290);

proc ark_flip_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('flip<uint64,3>', ark_flip_uint_3, 'ManipulationMsg', 290);

proc ark_flip_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('flip<uint64,4>', ark_flip_uint_4, 'ManipulationMsg', 290);

proc ark_flip_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('flip<uint8,1>', ark_flip_uint8_1, 'ManipulationMsg', 290);

proc ark_flip_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('flip<uint8,2>', ark_flip_uint8_2, 'ManipulationMsg', 290);

proc ark_flip_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('flip<uint8,3>', ark_flip_uint8_3, 'ManipulationMsg', 290);

proc ark_flip_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('flip<uint8,4>', ark_flip_uint8_4, 'ManipulationMsg', 290);

proc ark_flip_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('flip<float64,1>', ark_flip_real_1, 'ManipulationMsg', 290);

proc ark_flip_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('flip<float64,2>', ark_flip_real_2, 'ManipulationMsg', 290);

proc ark_flip_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('flip<float64,3>', ark_flip_real_3, 'ManipulationMsg', 290);

proc ark_flip_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('flip<float64,4>', ark_flip_real_4, 'ManipulationMsg', 290);

proc ark_flip_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('flip<bool,1>', ark_flip_bool_1, 'ManipulationMsg', 290);

proc ark_flip_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('flip<bool,2>', ark_flip_bool_2, 'ManipulationMsg', 290);

proc ark_flip_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('flip<bool,3>', ark_flip_bool_3, 'ManipulationMsg', 290);

proc ark_flip_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('flip<bool,4>', ark_flip_bool_4, 'ManipulationMsg', 290);

proc ark_flip_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('flip<bigint,1>', ark_flip_bigint_1, 'ManipulationMsg', 290);

proc ark_flip_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('flip<bigint,2>', ark_flip_bigint_2, 'ManipulationMsg', 290);

proc ark_flip_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('flip<bigint,3>', ark_flip_bigint_3, 'ManipulationMsg', 290);

proc ark_flip_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('flip<bigint,4>', ark_flip_bigint_4, 'ManipulationMsg', 290);

proc ark_flipAll_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('flipAll<int64,1>', ark_flipAll_int_1, 'ManipulationMsg', 358);

proc ark_flipAll_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('flipAll<int64,2>', ark_flipAll_int_2, 'ManipulationMsg', 358);

proc ark_flipAll_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('flipAll<int64,3>', ark_flipAll_int_3, 'ManipulationMsg', 358);

proc ark_flipAll_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('flipAll<int64,4>', ark_flipAll_int_4, 'ManipulationMsg', 358);

proc ark_flipAll_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('flipAll<uint64,1>', ark_flipAll_uint_1, 'ManipulationMsg', 358);

proc ark_flipAll_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('flipAll<uint64,2>', ark_flipAll_uint_2, 'ManipulationMsg', 358);

proc ark_flipAll_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('flipAll<uint64,3>', ark_flipAll_uint_3, 'ManipulationMsg', 358);

proc ark_flipAll_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('flipAll<uint64,4>', ark_flipAll_uint_4, 'ManipulationMsg', 358);

proc ark_flipAll_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('flipAll<uint8,1>', ark_flipAll_uint8_1, 'ManipulationMsg', 358);

proc ark_flipAll_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('flipAll<uint8,2>', ark_flipAll_uint8_2, 'ManipulationMsg', 358);

proc ark_flipAll_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('flipAll<uint8,3>', ark_flipAll_uint8_3, 'ManipulationMsg', 358);

proc ark_flipAll_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('flipAll<uint8,4>', ark_flipAll_uint8_4, 'ManipulationMsg', 358);

proc ark_flipAll_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('flipAll<float64,1>', ark_flipAll_real_1, 'ManipulationMsg', 358);

proc ark_flipAll_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('flipAll<float64,2>', ark_flipAll_real_2, 'ManipulationMsg', 358);

proc ark_flipAll_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('flipAll<float64,3>', ark_flipAll_real_3, 'ManipulationMsg', 358);

proc ark_flipAll_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('flipAll<float64,4>', ark_flipAll_real_4, 'ManipulationMsg', 358);

proc ark_flipAll_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('flipAll<bool,1>', ark_flipAll_bool_1, 'ManipulationMsg', 358);

proc ark_flipAll_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('flipAll<bool,2>', ark_flipAll_bool_2, 'ManipulationMsg', 358);

proc ark_flipAll_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('flipAll<bool,3>', ark_flipAll_bool_3, 'ManipulationMsg', 358);

proc ark_flipAll_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('flipAll<bool,4>', ark_flipAll_bool_4, 'ManipulationMsg', 358);

proc ark_flipAll_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('flipAll<bigint,1>', ark_flipAll_bigint_1, 'ManipulationMsg', 358);

proc ark_flipAll_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('flipAll<bigint,2>', ark_flipAll_bigint_2, 'ManipulationMsg', 358);

proc ark_flipAll_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('flipAll<bigint,3>', ark_flipAll_bigint_3, 'ManipulationMsg', 358);

proc ark_flipAll_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('flipAll<bigint,4>', ark_flipAll_bigint_4, 'ManipulationMsg', 358);

proc ark_permuteDims_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('permuteDims<int64,1>', ark_permuteDims_int_1, 'ManipulationMsg', 389);

proc ark_permuteDims_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('permuteDims<int64,2>', ark_permuteDims_int_2, 'ManipulationMsg', 389);

proc ark_permuteDims_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('permuteDims<int64,3>', ark_permuteDims_int_3, 'ManipulationMsg', 389);

proc ark_permuteDims_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('permuteDims<int64,4>', ark_permuteDims_int_4, 'ManipulationMsg', 389);

proc ark_permuteDims_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('permuteDims<uint64,1>', ark_permuteDims_uint_1, 'ManipulationMsg', 389);

proc ark_permuteDims_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('permuteDims<uint64,2>', ark_permuteDims_uint_2, 'ManipulationMsg', 389);

proc ark_permuteDims_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('permuteDims<uint64,3>', ark_permuteDims_uint_3, 'ManipulationMsg', 389);

proc ark_permuteDims_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('permuteDims<uint64,4>', ark_permuteDims_uint_4, 'ManipulationMsg', 389);

proc ark_permuteDims_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('permuteDims<uint8,1>', ark_permuteDims_uint8_1, 'ManipulationMsg', 389);

proc ark_permuteDims_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('permuteDims<uint8,2>', ark_permuteDims_uint8_2, 'ManipulationMsg', 389);

proc ark_permuteDims_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('permuteDims<uint8,3>', ark_permuteDims_uint8_3, 'ManipulationMsg', 389);

proc ark_permuteDims_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('permuteDims<uint8,4>', ark_permuteDims_uint8_4, 'ManipulationMsg', 389);

proc ark_permuteDims_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('permuteDims<float64,1>', ark_permuteDims_real_1, 'ManipulationMsg', 389);

proc ark_permuteDims_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('permuteDims<float64,2>', ark_permuteDims_real_2, 'ManipulationMsg', 389);

proc ark_permuteDims_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('permuteDims<float64,3>', ark_permuteDims_real_3, 'ManipulationMsg', 389);

proc ark_permuteDims_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('permuteDims<float64,4>', ark_permuteDims_real_4, 'ManipulationMsg', 389);

proc ark_permuteDims_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('permuteDims<bool,1>', ark_permuteDims_bool_1, 'ManipulationMsg', 389);

proc ark_permuteDims_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('permuteDims<bool,2>', ark_permuteDims_bool_2, 'ManipulationMsg', 389);

proc ark_permuteDims_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('permuteDims<bool,3>', ark_permuteDims_bool_3, 'ManipulationMsg', 389);

proc ark_permuteDims_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('permuteDims<bool,4>', ark_permuteDims_bool_4, 'ManipulationMsg', 389);

proc ark_permuteDims_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('permuteDims<bigint,1>', ark_permuteDims_bigint_1, 'ManipulationMsg', 389);

proc ark_permuteDims_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('permuteDims<bigint,2>', ark_permuteDims_bigint_2, 'ManipulationMsg', 389);

proc ark_permuteDims_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('permuteDims<bigint,3>', ark_permuteDims_bigint_3, 'ManipulationMsg', 389);

proc ark_permuteDims_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('permuteDims<bigint,4>', ark_permuteDims_bigint_4, 'ManipulationMsg', 389);

proc ark_reshape_int_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<int64,1,1>', ark_reshape_int_1_1, 'ManipulationMsg', 439);

proc ark_reshape_int_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=2);
registerFunction('reshape<int64,1,2>', ark_reshape_int_1_2, 'ManipulationMsg', 439);

proc ark_reshape_int_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=3);
registerFunction('reshape<int64,1,3>', ark_reshape_int_1_3, 'ManipulationMsg', 439);

proc ark_reshape_int_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=4);
registerFunction('reshape<int64,1,4>', ark_reshape_int_1_4, 'ManipulationMsg', 439);

proc ark_reshape_int_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=1);
registerFunction('reshape<int64,2,1>', ark_reshape_int_2_1, 'ManipulationMsg', 439);

proc ark_reshape_int_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=2);
registerFunction('reshape<int64,2,2>', ark_reshape_int_2_2, 'ManipulationMsg', 439);

proc ark_reshape_int_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=3);
registerFunction('reshape<int64,2,3>', ark_reshape_int_2_3, 'ManipulationMsg', 439);

proc ark_reshape_int_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=4);
registerFunction('reshape<int64,2,4>', ark_reshape_int_2_4, 'ManipulationMsg', 439);

proc ark_reshape_int_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=1);
registerFunction('reshape<int64,3,1>', ark_reshape_int_3_1, 'ManipulationMsg', 439);

proc ark_reshape_int_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=2);
registerFunction('reshape<int64,3,2>', ark_reshape_int_3_2, 'ManipulationMsg', 439);

proc ark_reshape_int_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=3);
registerFunction('reshape<int64,3,3>', ark_reshape_int_3_3, 'ManipulationMsg', 439);

proc ark_reshape_int_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=4);
registerFunction('reshape<int64,3,4>', ark_reshape_int_3_4, 'ManipulationMsg', 439);

proc ark_reshape_int_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=4, array_nd_out=1);
registerFunction('reshape<int64,4,1>', ark_reshape_int_4_1, 'ManipulationMsg', 439);

proc ark_reshape_int_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=4, array_nd_out=2);
registerFunction('reshape<int64,4,2>', ark_reshape_int_4_2, 'ManipulationMsg', 439);

proc ark_reshape_int_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=4, array_nd_out=3);
registerFunction('reshape<int64,4,3>', ark_reshape_int_4_3, 'ManipulationMsg', 439);

proc ark_reshape_int_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=4, array_nd_out=4);
registerFunction('reshape<int64,4,4>', ark_reshape_int_4_4, 'ManipulationMsg', 439);

proc ark_reshape_uint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<uint64,1,1>', ark_reshape_uint_1_1, 'ManipulationMsg', 439);

proc ark_reshape_uint_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=2);
registerFunction('reshape<uint64,1,2>', ark_reshape_uint_1_2, 'ManipulationMsg', 439);

proc ark_reshape_uint_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=3);
registerFunction('reshape<uint64,1,3>', ark_reshape_uint_1_3, 'ManipulationMsg', 439);

proc ark_reshape_uint_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=4);
registerFunction('reshape<uint64,1,4>', ark_reshape_uint_1_4, 'ManipulationMsg', 439);

proc ark_reshape_uint_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=2, array_nd_out=1);
registerFunction('reshape<uint64,2,1>', ark_reshape_uint_2_1, 'ManipulationMsg', 439);

proc ark_reshape_uint_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=2, array_nd_out=2);
registerFunction('reshape<uint64,2,2>', ark_reshape_uint_2_2, 'ManipulationMsg', 439);

proc ark_reshape_uint_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=2, array_nd_out=3);
registerFunction('reshape<uint64,2,3>', ark_reshape_uint_2_3, 'ManipulationMsg', 439);

proc ark_reshape_uint_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=2, array_nd_out=4);
registerFunction('reshape<uint64,2,4>', ark_reshape_uint_2_4, 'ManipulationMsg', 439);

proc ark_reshape_uint_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=3, array_nd_out=1);
registerFunction('reshape<uint64,3,1>', ark_reshape_uint_3_1, 'ManipulationMsg', 439);

proc ark_reshape_uint_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=3, array_nd_out=2);
registerFunction('reshape<uint64,3,2>', ark_reshape_uint_3_2, 'ManipulationMsg', 439);

proc ark_reshape_uint_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=3, array_nd_out=3);
registerFunction('reshape<uint64,3,3>', ark_reshape_uint_3_3, 'ManipulationMsg', 439);

proc ark_reshape_uint_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=3, array_nd_out=4);
registerFunction('reshape<uint64,3,4>', ark_reshape_uint_3_4, 'ManipulationMsg', 439);

proc ark_reshape_uint_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=4, array_nd_out=1);
registerFunction('reshape<uint64,4,1>', ark_reshape_uint_4_1, 'ManipulationMsg', 439);

proc ark_reshape_uint_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=4, array_nd_out=2);
registerFunction('reshape<uint64,4,2>', ark_reshape_uint_4_2, 'ManipulationMsg', 439);

proc ark_reshape_uint_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=4, array_nd_out=3);
registerFunction('reshape<uint64,4,3>', ark_reshape_uint_4_3, 'ManipulationMsg', 439);

proc ark_reshape_uint_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=4, array_nd_out=4);
registerFunction('reshape<uint64,4,4>', ark_reshape_uint_4_4, 'ManipulationMsg', 439);

proc ark_reshape_uint8_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=1);
registerFunction('reshape<uint8,1,1>', ark_reshape_uint8_1_1, 'ManipulationMsg', 439);

proc ark_reshape_uint8_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=2);
registerFunction('reshape<uint8,1,2>', ark_reshape_uint8_1_2, 'ManipulationMsg', 439);

proc ark_reshape_uint8_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=3);
registerFunction('reshape<uint8,1,3>', ark_reshape_uint8_1_3, 'ManipulationMsg', 439);

proc ark_reshape_uint8_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=4);
registerFunction('reshape<uint8,1,4>', ark_reshape_uint8_1_4, 'ManipulationMsg', 439);

proc ark_reshape_uint8_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=2, array_nd_out=1);
registerFunction('reshape<uint8,2,1>', ark_reshape_uint8_2_1, 'ManipulationMsg', 439);

proc ark_reshape_uint8_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=2, array_nd_out=2);
registerFunction('reshape<uint8,2,2>', ark_reshape_uint8_2_2, 'ManipulationMsg', 439);

proc ark_reshape_uint8_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=2, array_nd_out=3);
registerFunction('reshape<uint8,2,3>', ark_reshape_uint8_2_3, 'ManipulationMsg', 439);

proc ark_reshape_uint8_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=2, array_nd_out=4);
registerFunction('reshape<uint8,2,4>', ark_reshape_uint8_2_4, 'ManipulationMsg', 439);

proc ark_reshape_uint8_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=3, array_nd_out=1);
registerFunction('reshape<uint8,3,1>', ark_reshape_uint8_3_1, 'ManipulationMsg', 439);

proc ark_reshape_uint8_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=3, array_nd_out=2);
registerFunction('reshape<uint8,3,2>', ark_reshape_uint8_3_2, 'ManipulationMsg', 439);

proc ark_reshape_uint8_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=3, array_nd_out=3);
registerFunction('reshape<uint8,3,3>', ark_reshape_uint8_3_3, 'ManipulationMsg', 439);

proc ark_reshape_uint8_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=3, array_nd_out=4);
registerFunction('reshape<uint8,3,4>', ark_reshape_uint8_3_4, 'ManipulationMsg', 439);

proc ark_reshape_uint8_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=4, array_nd_out=1);
registerFunction('reshape<uint8,4,1>', ark_reshape_uint8_4_1, 'ManipulationMsg', 439);

proc ark_reshape_uint8_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=4, array_nd_out=2);
registerFunction('reshape<uint8,4,2>', ark_reshape_uint8_4_2, 'ManipulationMsg', 439);

proc ark_reshape_uint8_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=4, array_nd_out=3);
registerFunction('reshape<uint8,4,3>', ark_reshape_uint8_4_3, 'ManipulationMsg', 439);

proc ark_reshape_uint8_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=4, array_nd_out=4);
registerFunction('reshape<uint8,4,4>', ark_reshape_uint8_4_4, 'ManipulationMsg', 439);

proc ark_reshape_real_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<float64,1,1>', ark_reshape_real_1_1, 'ManipulationMsg', 439);

proc ark_reshape_real_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=2);
registerFunction('reshape<float64,1,2>', ark_reshape_real_1_2, 'ManipulationMsg', 439);

proc ark_reshape_real_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=3);
registerFunction('reshape<float64,1,3>', ark_reshape_real_1_3, 'ManipulationMsg', 439);

proc ark_reshape_real_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=4);
registerFunction('reshape<float64,1,4>', ark_reshape_real_1_4, 'ManipulationMsg', 439);

proc ark_reshape_real_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=1);
registerFunction('reshape<float64,2,1>', ark_reshape_real_2_1, 'ManipulationMsg', 439);

proc ark_reshape_real_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=2);
registerFunction('reshape<float64,2,2>', ark_reshape_real_2_2, 'ManipulationMsg', 439);

proc ark_reshape_real_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=3);
registerFunction('reshape<float64,2,3>', ark_reshape_real_2_3, 'ManipulationMsg', 439);

proc ark_reshape_real_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=4);
registerFunction('reshape<float64,2,4>', ark_reshape_real_2_4, 'ManipulationMsg', 439);

proc ark_reshape_real_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=1);
registerFunction('reshape<float64,3,1>', ark_reshape_real_3_1, 'ManipulationMsg', 439);

proc ark_reshape_real_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=2);
registerFunction('reshape<float64,3,2>', ark_reshape_real_3_2, 'ManipulationMsg', 439);

proc ark_reshape_real_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=3);
registerFunction('reshape<float64,3,3>', ark_reshape_real_3_3, 'ManipulationMsg', 439);

proc ark_reshape_real_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=4);
registerFunction('reshape<float64,3,4>', ark_reshape_real_3_4, 'ManipulationMsg', 439);

proc ark_reshape_real_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=4, array_nd_out=1);
registerFunction('reshape<float64,4,1>', ark_reshape_real_4_1, 'ManipulationMsg', 439);

proc ark_reshape_real_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=4, array_nd_out=2);
registerFunction('reshape<float64,4,2>', ark_reshape_real_4_2, 'ManipulationMsg', 439);

proc ark_reshape_real_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=4, array_nd_out=3);
registerFunction('reshape<float64,4,3>', ark_reshape_real_4_3, 'ManipulationMsg', 439);

proc ark_reshape_real_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=4, array_nd_out=4);
registerFunction('reshape<float64,4,4>', ark_reshape_real_4_4, 'ManipulationMsg', 439);

proc ark_reshape_bool_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<bool,1,1>', ark_reshape_bool_1_1, 'ManipulationMsg', 439);

proc ark_reshape_bool_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=2);
registerFunction('reshape<bool,1,2>', ark_reshape_bool_1_2, 'ManipulationMsg', 439);

proc ark_reshape_bool_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=3);
registerFunction('reshape<bool,1,3>', ark_reshape_bool_1_3, 'ManipulationMsg', 439);

proc ark_reshape_bool_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=4);
registerFunction('reshape<bool,1,4>', ark_reshape_bool_1_4, 'ManipulationMsg', 439);

proc ark_reshape_bool_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=2, array_nd_out=1);
registerFunction('reshape<bool,2,1>', ark_reshape_bool_2_1, 'ManipulationMsg', 439);

proc ark_reshape_bool_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=2, array_nd_out=2);
registerFunction('reshape<bool,2,2>', ark_reshape_bool_2_2, 'ManipulationMsg', 439);

proc ark_reshape_bool_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=2, array_nd_out=3);
registerFunction('reshape<bool,2,3>', ark_reshape_bool_2_3, 'ManipulationMsg', 439);

proc ark_reshape_bool_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=2, array_nd_out=4);
registerFunction('reshape<bool,2,4>', ark_reshape_bool_2_4, 'ManipulationMsg', 439);

proc ark_reshape_bool_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=3, array_nd_out=1);
registerFunction('reshape<bool,3,1>', ark_reshape_bool_3_1, 'ManipulationMsg', 439);

proc ark_reshape_bool_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=3, array_nd_out=2);
registerFunction('reshape<bool,3,2>', ark_reshape_bool_3_2, 'ManipulationMsg', 439);

proc ark_reshape_bool_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=3, array_nd_out=3);
registerFunction('reshape<bool,3,3>', ark_reshape_bool_3_3, 'ManipulationMsg', 439);

proc ark_reshape_bool_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=3, array_nd_out=4);
registerFunction('reshape<bool,3,4>', ark_reshape_bool_3_4, 'ManipulationMsg', 439);

proc ark_reshape_bool_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=4, array_nd_out=1);
registerFunction('reshape<bool,4,1>', ark_reshape_bool_4_1, 'ManipulationMsg', 439);

proc ark_reshape_bool_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=4, array_nd_out=2);
registerFunction('reshape<bool,4,2>', ark_reshape_bool_4_2, 'ManipulationMsg', 439);

proc ark_reshape_bool_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=4, array_nd_out=3);
registerFunction('reshape<bool,4,3>', ark_reshape_bool_4_3, 'ManipulationMsg', 439);

proc ark_reshape_bool_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=4, array_nd_out=4);
registerFunction('reshape<bool,4,4>', ark_reshape_bool_4_4, 'ManipulationMsg', 439);

proc ark_reshape_bigint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<bigint,1,1>', ark_reshape_bigint_1_1, 'ManipulationMsg', 439);

proc ark_reshape_bigint_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=2);
registerFunction('reshape<bigint,1,2>', ark_reshape_bigint_1_2, 'ManipulationMsg', 439);

proc ark_reshape_bigint_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=3);
registerFunction('reshape<bigint,1,3>', ark_reshape_bigint_1_3, 'ManipulationMsg', 439);

proc ark_reshape_bigint_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=4);
registerFunction('reshape<bigint,1,4>', ark_reshape_bigint_1_4, 'ManipulationMsg', 439);

proc ark_reshape_bigint_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=2, array_nd_out=1);
registerFunction('reshape<bigint,2,1>', ark_reshape_bigint_2_1, 'ManipulationMsg', 439);

proc ark_reshape_bigint_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=2, array_nd_out=2);
registerFunction('reshape<bigint,2,2>', ark_reshape_bigint_2_2, 'ManipulationMsg', 439);

proc ark_reshape_bigint_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=2, array_nd_out=3);
registerFunction('reshape<bigint,2,3>', ark_reshape_bigint_2_3, 'ManipulationMsg', 439);

proc ark_reshape_bigint_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=2, array_nd_out=4);
registerFunction('reshape<bigint,2,4>', ark_reshape_bigint_2_4, 'ManipulationMsg', 439);

proc ark_reshape_bigint_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=3, array_nd_out=1);
registerFunction('reshape<bigint,3,1>', ark_reshape_bigint_3_1, 'ManipulationMsg', 439);

proc ark_reshape_bigint_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=3, array_nd_out=2);
registerFunction('reshape<bigint,3,2>', ark_reshape_bigint_3_2, 'ManipulationMsg', 439);

proc ark_reshape_bigint_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=3, array_nd_out=3);
registerFunction('reshape<bigint,3,3>', ark_reshape_bigint_3_3, 'ManipulationMsg', 439);

proc ark_reshape_bigint_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=3, array_nd_out=4);
registerFunction('reshape<bigint,3,4>', ark_reshape_bigint_3_4, 'ManipulationMsg', 439);

proc ark_reshape_bigint_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=4, array_nd_out=1);
registerFunction('reshape<bigint,4,1>', ark_reshape_bigint_4_1, 'ManipulationMsg', 439);

proc ark_reshape_bigint_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=4, array_nd_out=2);
registerFunction('reshape<bigint,4,2>', ark_reshape_bigint_4_2, 'ManipulationMsg', 439);

proc ark_reshape_bigint_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=4, array_nd_out=3);
registerFunction('reshape<bigint,4,3>', ark_reshape_bigint_4_3, 'ManipulationMsg', 439);

proc ark_reshape_bigint_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=4, array_nd_out=4);
registerFunction('reshape<bigint,4,4>', ark_reshape_bigint_4_4, 'ManipulationMsg', 439);

proc ark_roll_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('roll<int64,1>', ark_roll_int_1, 'ManipulationMsg', 521);

proc ark_roll_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('roll<int64,2>', ark_roll_int_2, 'ManipulationMsg', 521);

proc ark_roll_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('roll<int64,3>', ark_roll_int_3, 'ManipulationMsg', 521);

proc ark_roll_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('roll<int64,4>', ark_roll_int_4, 'ManipulationMsg', 521);

proc ark_roll_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('roll<uint64,1>', ark_roll_uint_1, 'ManipulationMsg', 521);

proc ark_roll_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('roll<uint64,2>', ark_roll_uint_2, 'ManipulationMsg', 521);

proc ark_roll_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('roll<uint64,3>', ark_roll_uint_3, 'ManipulationMsg', 521);

proc ark_roll_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('roll<uint64,4>', ark_roll_uint_4, 'ManipulationMsg', 521);

proc ark_roll_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('roll<uint8,1>', ark_roll_uint8_1, 'ManipulationMsg', 521);

proc ark_roll_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('roll<uint8,2>', ark_roll_uint8_2, 'ManipulationMsg', 521);

proc ark_roll_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('roll<uint8,3>', ark_roll_uint8_3, 'ManipulationMsg', 521);

proc ark_roll_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('roll<uint8,4>', ark_roll_uint8_4, 'ManipulationMsg', 521);

proc ark_roll_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('roll<float64,1>', ark_roll_real_1, 'ManipulationMsg', 521);

proc ark_roll_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('roll<float64,2>', ark_roll_real_2, 'ManipulationMsg', 521);

proc ark_roll_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('roll<float64,3>', ark_roll_real_3, 'ManipulationMsg', 521);

proc ark_roll_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('roll<float64,4>', ark_roll_real_4, 'ManipulationMsg', 521);

proc ark_roll_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('roll<bool,1>', ark_roll_bool_1, 'ManipulationMsg', 521);

proc ark_roll_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('roll<bool,2>', ark_roll_bool_2, 'ManipulationMsg', 521);

proc ark_roll_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('roll<bool,3>', ark_roll_bool_3, 'ManipulationMsg', 521);

proc ark_roll_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('roll<bool,4>', ark_roll_bool_4, 'ManipulationMsg', 521);

proc ark_roll_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('roll<bigint,1>', ark_roll_bigint_1, 'ManipulationMsg', 521);

proc ark_roll_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('roll<bigint,2>', ark_roll_bigint_2, 'ManipulationMsg', 521);

proc ark_roll_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('roll<bigint,3>', ark_roll_bigint_3, 'ManipulationMsg', 521);

proc ark_roll_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('roll<bigint,4>', ark_roll_bigint_4, 'ManipulationMsg', 521);

proc ark_rollFlattened_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('rollFlattened<int64,1>', ark_rollFlattened_int_1, 'ManipulationMsg', 586);

proc ark_rollFlattened_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('rollFlattened<int64,2>', ark_rollFlattened_int_2, 'ManipulationMsg', 586);

proc ark_rollFlattened_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('rollFlattened<int64,3>', ark_rollFlattened_int_3, 'ManipulationMsg', 586);

proc ark_rollFlattened_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('rollFlattened<int64,4>', ark_rollFlattened_int_4, 'ManipulationMsg', 586);

proc ark_rollFlattened_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('rollFlattened<uint64,1>', ark_rollFlattened_uint_1, 'ManipulationMsg', 586);

proc ark_rollFlattened_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('rollFlattened<uint64,2>', ark_rollFlattened_uint_2, 'ManipulationMsg', 586);

proc ark_rollFlattened_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('rollFlattened<uint64,3>', ark_rollFlattened_uint_3, 'ManipulationMsg', 586);

proc ark_rollFlattened_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('rollFlattened<uint64,4>', ark_rollFlattened_uint_4, 'ManipulationMsg', 586);

proc ark_rollFlattened_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('rollFlattened<uint8,1>', ark_rollFlattened_uint8_1, 'ManipulationMsg', 586);

proc ark_rollFlattened_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('rollFlattened<uint8,2>', ark_rollFlattened_uint8_2, 'ManipulationMsg', 586);

proc ark_rollFlattened_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('rollFlattened<uint8,3>', ark_rollFlattened_uint8_3, 'ManipulationMsg', 586);

proc ark_rollFlattened_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('rollFlattened<uint8,4>', ark_rollFlattened_uint8_4, 'ManipulationMsg', 586);

proc ark_rollFlattened_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('rollFlattened<float64,1>', ark_rollFlattened_real_1, 'ManipulationMsg', 586);

proc ark_rollFlattened_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('rollFlattened<float64,2>', ark_rollFlattened_real_2, 'ManipulationMsg', 586);

proc ark_rollFlattened_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('rollFlattened<float64,3>', ark_rollFlattened_real_3, 'ManipulationMsg', 586);

proc ark_rollFlattened_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('rollFlattened<float64,4>', ark_rollFlattened_real_4, 'ManipulationMsg', 586);

proc ark_rollFlattened_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('rollFlattened<bool,1>', ark_rollFlattened_bool_1, 'ManipulationMsg', 586);

proc ark_rollFlattened_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('rollFlattened<bool,2>', ark_rollFlattened_bool_2, 'ManipulationMsg', 586);

proc ark_rollFlattened_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('rollFlattened<bool,3>', ark_rollFlattened_bool_3, 'ManipulationMsg', 586);

proc ark_rollFlattened_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('rollFlattened<bool,4>', ark_rollFlattened_bool_4, 'ManipulationMsg', 586);

proc ark_rollFlattened_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('rollFlattened<bigint,1>', ark_rollFlattened_bigint_1, 'ManipulationMsg', 586);

proc ark_rollFlattened_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('rollFlattened<bigint,2>', ark_rollFlattened_bigint_2, 'ManipulationMsg', 586);

proc ark_rollFlattened_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('rollFlattened<bigint,3>', ark_rollFlattened_bigint_3, 'ManipulationMsg', 586);

proc ark_rollFlattened_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('rollFlattened<bigint,4>', ark_rollFlattened_bigint_4, 'ManipulationMsg', 586);

proc ark_squeeze_int_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<int64,1,1>', ark_squeeze_int_1_1, 'ManipulationMsg', 607);

proc ark_squeeze_int_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=2);
registerFunction('squeeze<int64,1,2>', ark_squeeze_int_1_2, 'ManipulationMsg', 607);

proc ark_squeeze_int_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=3);
registerFunction('squeeze<int64,1,3>', ark_squeeze_int_1_3, 'ManipulationMsg', 607);

proc ark_squeeze_int_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=4);
registerFunction('squeeze<int64,1,4>', ark_squeeze_int_1_4, 'ManipulationMsg', 607);

proc ark_squeeze_int_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=1);
registerFunction('squeeze<int64,2,1>', ark_squeeze_int_2_1, 'ManipulationMsg', 607);

proc ark_squeeze_int_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=2);
registerFunction('squeeze<int64,2,2>', ark_squeeze_int_2_2, 'ManipulationMsg', 607);

proc ark_squeeze_int_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=3);
registerFunction('squeeze<int64,2,3>', ark_squeeze_int_2_3, 'ManipulationMsg', 607);

proc ark_squeeze_int_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=4);
registerFunction('squeeze<int64,2,4>', ark_squeeze_int_2_4, 'ManipulationMsg', 607);

proc ark_squeeze_int_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=1);
registerFunction('squeeze<int64,3,1>', ark_squeeze_int_3_1, 'ManipulationMsg', 607);

proc ark_squeeze_int_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=2);
registerFunction('squeeze<int64,3,2>', ark_squeeze_int_3_2, 'ManipulationMsg', 607);

proc ark_squeeze_int_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=3);
registerFunction('squeeze<int64,3,3>', ark_squeeze_int_3_3, 'ManipulationMsg', 607);

proc ark_squeeze_int_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=4);
registerFunction('squeeze<int64,3,4>', ark_squeeze_int_3_4, 'ManipulationMsg', 607);

proc ark_squeeze_int_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=4, array_nd_out=1);
registerFunction('squeeze<int64,4,1>', ark_squeeze_int_4_1, 'ManipulationMsg', 607);

proc ark_squeeze_int_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=4, array_nd_out=2);
registerFunction('squeeze<int64,4,2>', ark_squeeze_int_4_2, 'ManipulationMsg', 607);

proc ark_squeeze_int_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=4, array_nd_out=3);
registerFunction('squeeze<int64,4,3>', ark_squeeze_int_4_3, 'ManipulationMsg', 607);

proc ark_squeeze_int_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=4, array_nd_out=4);
registerFunction('squeeze<int64,4,4>', ark_squeeze_int_4_4, 'ManipulationMsg', 607);

proc ark_squeeze_uint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<uint64,1,1>', ark_squeeze_uint_1_1, 'ManipulationMsg', 607);

proc ark_squeeze_uint_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=2);
registerFunction('squeeze<uint64,1,2>', ark_squeeze_uint_1_2, 'ManipulationMsg', 607);

proc ark_squeeze_uint_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=3);
registerFunction('squeeze<uint64,1,3>', ark_squeeze_uint_1_3, 'ManipulationMsg', 607);

proc ark_squeeze_uint_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=4);
registerFunction('squeeze<uint64,1,4>', ark_squeeze_uint_1_4, 'ManipulationMsg', 607);

proc ark_squeeze_uint_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=2, array_nd_out=1);
registerFunction('squeeze<uint64,2,1>', ark_squeeze_uint_2_1, 'ManipulationMsg', 607);

proc ark_squeeze_uint_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=2, array_nd_out=2);
registerFunction('squeeze<uint64,2,2>', ark_squeeze_uint_2_2, 'ManipulationMsg', 607);

proc ark_squeeze_uint_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=2, array_nd_out=3);
registerFunction('squeeze<uint64,2,3>', ark_squeeze_uint_2_3, 'ManipulationMsg', 607);

proc ark_squeeze_uint_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=2, array_nd_out=4);
registerFunction('squeeze<uint64,2,4>', ark_squeeze_uint_2_4, 'ManipulationMsg', 607);

proc ark_squeeze_uint_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=3, array_nd_out=1);
registerFunction('squeeze<uint64,3,1>', ark_squeeze_uint_3_1, 'ManipulationMsg', 607);

proc ark_squeeze_uint_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=3, array_nd_out=2);
registerFunction('squeeze<uint64,3,2>', ark_squeeze_uint_3_2, 'ManipulationMsg', 607);

proc ark_squeeze_uint_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=3, array_nd_out=3);
registerFunction('squeeze<uint64,3,3>', ark_squeeze_uint_3_3, 'ManipulationMsg', 607);

proc ark_squeeze_uint_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=3, array_nd_out=4);
registerFunction('squeeze<uint64,3,4>', ark_squeeze_uint_3_4, 'ManipulationMsg', 607);

proc ark_squeeze_uint_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=4, array_nd_out=1);
registerFunction('squeeze<uint64,4,1>', ark_squeeze_uint_4_1, 'ManipulationMsg', 607);

proc ark_squeeze_uint_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=4, array_nd_out=2);
registerFunction('squeeze<uint64,4,2>', ark_squeeze_uint_4_2, 'ManipulationMsg', 607);

proc ark_squeeze_uint_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=4, array_nd_out=3);
registerFunction('squeeze<uint64,4,3>', ark_squeeze_uint_4_3, 'ManipulationMsg', 607);

proc ark_squeeze_uint_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=4, array_nd_out=4);
registerFunction('squeeze<uint64,4,4>', ark_squeeze_uint_4_4, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<uint8,1,1>', ark_squeeze_uint8_1_1, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=2);
registerFunction('squeeze<uint8,1,2>', ark_squeeze_uint8_1_2, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=3);
registerFunction('squeeze<uint8,1,3>', ark_squeeze_uint8_1_3, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=1, array_nd_out=4);
registerFunction('squeeze<uint8,1,4>', ark_squeeze_uint8_1_4, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=2, array_nd_out=1);
registerFunction('squeeze<uint8,2,1>', ark_squeeze_uint8_2_1, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=2, array_nd_out=2);
registerFunction('squeeze<uint8,2,2>', ark_squeeze_uint8_2_2, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=2, array_nd_out=3);
registerFunction('squeeze<uint8,2,3>', ark_squeeze_uint8_2_3, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=2, array_nd_out=4);
registerFunction('squeeze<uint8,2,4>', ark_squeeze_uint8_2_4, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=3, array_nd_out=1);
registerFunction('squeeze<uint8,3,1>', ark_squeeze_uint8_3_1, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=3, array_nd_out=2);
registerFunction('squeeze<uint8,3,2>', ark_squeeze_uint8_3_2, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=3, array_nd_out=3);
registerFunction('squeeze<uint8,3,3>', ark_squeeze_uint8_3_3, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=3, array_nd_out=4);
registerFunction('squeeze<uint8,3,4>', ark_squeeze_uint8_3_4, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=4, array_nd_out=1);
registerFunction('squeeze<uint8,4,1>', ark_squeeze_uint8_4_1, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=4, array_nd_out=2);
registerFunction('squeeze<uint8,4,2>', ark_squeeze_uint8_4_2, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=4, array_nd_out=3);
registerFunction('squeeze<uint8,4,3>', ark_squeeze_uint8_4_3, 'ManipulationMsg', 607);

proc ark_squeeze_uint8_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd_in=4, array_nd_out=4);
registerFunction('squeeze<uint8,4,4>', ark_squeeze_uint8_4_4, 'ManipulationMsg', 607);

proc ark_squeeze_real_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<float64,1,1>', ark_squeeze_real_1_1, 'ManipulationMsg', 607);

proc ark_squeeze_real_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=2);
registerFunction('squeeze<float64,1,2>', ark_squeeze_real_1_2, 'ManipulationMsg', 607);

proc ark_squeeze_real_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=3);
registerFunction('squeeze<float64,1,3>', ark_squeeze_real_1_3, 'ManipulationMsg', 607);

proc ark_squeeze_real_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=4);
registerFunction('squeeze<float64,1,4>', ark_squeeze_real_1_4, 'ManipulationMsg', 607);

proc ark_squeeze_real_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=1);
registerFunction('squeeze<float64,2,1>', ark_squeeze_real_2_1, 'ManipulationMsg', 607);

proc ark_squeeze_real_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=2);
registerFunction('squeeze<float64,2,2>', ark_squeeze_real_2_2, 'ManipulationMsg', 607);

proc ark_squeeze_real_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=3);
registerFunction('squeeze<float64,2,3>', ark_squeeze_real_2_3, 'ManipulationMsg', 607);

proc ark_squeeze_real_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=4);
registerFunction('squeeze<float64,2,4>', ark_squeeze_real_2_4, 'ManipulationMsg', 607);

proc ark_squeeze_real_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=1);
registerFunction('squeeze<float64,3,1>', ark_squeeze_real_3_1, 'ManipulationMsg', 607);

proc ark_squeeze_real_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=2);
registerFunction('squeeze<float64,3,2>', ark_squeeze_real_3_2, 'ManipulationMsg', 607);

proc ark_squeeze_real_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=3);
registerFunction('squeeze<float64,3,3>', ark_squeeze_real_3_3, 'ManipulationMsg', 607);

proc ark_squeeze_real_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=4);
registerFunction('squeeze<float64,3,4>', ark_squeeze_real_3_4, 'ManipulationMsg', 607);

proc ark_squeeze_real_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=4, array_nd_out=1);
registerFunction('squeeze<float64,4,1>', ark_squeeze_real_4_1, 'ManipulationMsg', 607);

proc ark_squeeze_real_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=4, array_nd_out=2);
registerFunction('squeeze<float64,4,2>', ark_squeeze_real_4_2, 'ManipulationMsg', 607);

proc ark_squeeze_real_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=4, array_nd_out=3);
registerFunction('squeeze<float64,4,3>', ark_squeeze_real_4_3, 'ManipulationMsg', 607);

proc ark_squeeze_real_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=4, array_nd_out=4);
registerFunction('squeeze<float64,4,4>', ark_squeeze_real_4_4, 'ManipulationMsg', 607);

proc ark_squeeze_bool_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<bool,1,1>', ark_squeeze_bool_1_1, 'ManipulationMsg', 607);

proc ark_squeeze_bool_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=2);
registerFunction('squeeze<bool,1,2>', ark_squeeze_bool_1_2, 'ManipulationMsg', 607);

proc ark_squeeze_bool_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=3);
registerFunction('squeeze<bool,1,3>', ark_squeeze_bool_1_3, 'ManipulationMsg', 607);

proc ark_squeeze_bool_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=4);
registerFunction('squeeze<bool,1,4>', ark_squeeze_bool_1_4, 'ManipulationMsg', 607);

proc ark_squeeze_bool_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=2, array_nd_out=1);
registerFunction('squeeze<bool,2,1>', ark_squeeze_bool_2_1, 'ManipulationMsg', 607);

proc ark_squeeze_bool_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=2, array_nd_out=2);
registerFunction('squeeze<bool,2,2>', ark_squeeze_bool_2_2, 'ManipulationMsg', 607);

proc ark_squeeze_bool_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=2, array_nd_out=3);
registerFunction('squeeze<bool,2,3>', ark_squeeze_bool_2_3, 'ManipulationMsg', 607);

proc ark_squeeze_bool_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=2, array_nd_out=4);
registerFunction('squeeze<bool,2,4>', ark_squeeze_bool_2_4, 'ManipulationMsg', 607);

proc ark_squeeze_bool_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=3, array_nd_out=1);
registerFunction('squeeze<bool,3,1>', ark_squeeze_bool_3_1, 'ManipulationMsg', 607);

proc ark_squeeze_bool_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=3, array_nd_out=2);
registerFunction('squeeze<bool,3,2>', ark_squeeze_bool_3_2, 'ManipulationMsg', 607);

proc ark_squeeze_bool_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=3, array_nd_out=3);
registerFunction('squeeze<bool,3,3>', ark_squeeze_bool_3_3, 'ManipulationMsg', 607);

proc ark_squeeze_bool_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=3, array_nd_out=4);
registerFunction('squeeze<bool,3,4>', ark_squeeze_bool_3_4, 'ManipulationMsg', 607);

proc ark_squeeze_bool_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=4, array_nd_out=1);
registerFunction('squeeze<bool,4,1>', ark_squeeze_bool_4_1, 'ManipulationMsg', 607);

proc ark_squeeze_bool_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=4, array_nd_out=2);
registerFunction('squeeze<bool,4,2>', ark_squeeze_bool_4_2, 'ManipulationMsg', 607);

proc ark_squeeze_bool_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=4, array_nd_out=3);
registerFunction('squeeze<bool,4,3>', ark_squeeze_bool_4_3, 'ManipulationMsg', 607);

proc ark_squeeze_bool_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=4, array_nd_out=4);
registerFunction('squeeze<bool,4,4>', ark_squeeze_bool_4_4, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<bigint,1,1>', ark_squeeze_bigint_1_1, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=2);
registerFunction('squeeze<bigint,1,2>', ark_squeeze_bigint_1_2, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=3);
registerFunction('squeeze<bigint,1,3>', ark_squeeze_bigint_1_3, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_1_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=1, array_nd_out=4);
registerFunction('squeeze<bigint,1,4>', ark_squeeze_bigint_1_4, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=2, array_nd_out=1);
registerFunction('squeeze<bigint,2,1>', ark_squeeze_bigint_2_1, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=2, array_nd_out=2);
registerFunction('squeeze<bigint,2,2>', ark_squeeze_bigint_2_2, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=2, array_nd_out=3);
registerFunction('squeeze<bigint,2,3>', ark_squeeze_bigint_2_3, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_2_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=2, array_nd_out=4);
registerFunction('squeeze<bigint,2,4>', ark_squeeze_bigint_2_4, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=3, array_nd_out=1);
registerFunction('squeeze<bigint,3,1>', ark_squeeze_bigint_3_1, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=3, array_nd_out=2);
registerFunction('squeeze<bigint,3,2>', ark_squeeze_bigint_3_2, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=3, array_nd_out=3);
registerFunction('squeeze<bigint,3,3>', ark_squeeze_bigint_3_3, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_3_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=3, array_nd_out=4);
registerFunction('squeeze<bigint,3,4>', ark_squeeze_bigint_3_4, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_4_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=4, array_nd_out=1);
registerFunction('squeeze<bigint,4,1>', ark_squeeze_bigint_4_1, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_4_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=4, array_nd_out=2);
registerFunction('squeeze<bigint,4,2>', ark_squeeze_bigint_4_2, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_4_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=4, array_nd_out=3);
registerFunction('squeeze<bigint,4,3>', ark_squeeze_bigint_4_3, 'ManipulationMsg', 607);

proc ark_squeeze_bigint_4_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd_in=4, array_nd_out=4);
registerFunction('squeeze<bigint,4,4>', ark_squeeze_bigint_4_4, 'ManipulationMsg', 607);

proc ark_stack_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('stack<int64,1>', ark_stack_int_1, 'ManipulationMsg', 686);

proc ark_stack_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('stack<int64,2>', ark_stack_int_2, 'ManipulationMsg', 686);

proc ark_stack_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('stack<int64,3>', ark_stack_int_3, 'ManipulationMsg', 686);

proc ark_stack_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('stack<int64,4>', ark_stack_int_4, 'ManipulationMsg', 686);

proc ark_stack_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('stack<uint64,1>', ark_stack_uint_1, 'ManipulationMsg', 686);

proc ark_stack_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('stack<uint64,2>', ark_stack_uint_2, 'ManipulationMsg', 686);

proc ark_stack_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('stack<uint64,3>', ark_stack_uint_3, 'ManipulationMsg', 686);

proc ark_stack_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('stack<uint64,4>', ark_stack_uint_4, 'ManipulationMsg', 686);

proc ark_stack_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('stack<uint8,1>', ark_stack_uint8_1, 'ManipulationMsg', 686);

proc ark_stack_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('stack<uint8,2>', ark_stack_uint8_2, 'ManipulationMsg', 686);

proc ark_stack_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('stack<uint8,3>', ark_stack_uint8_3, 'ManipulationMsg', 686);

proc ark_stack_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('stack<uint8,4>', ark_stack_uint8_4, 'ManipulationMsg', 686);

proc ark_stack_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('stack<float64,1>', ark_stack_real_1, 'ManipulationMsg', 686);

proc ark_stack_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('stack<float64,2>', ark_stack_real_2, 'ManipulationMsg', 686);

proc ark_stack_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('stack<float64,3>', ark_stack_real_3, 'ManipulationMsg', 686);

proc ark_stack_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('stack<float64,4>', ark_stack_real_4, 'ManipulationMsg', 686);

proc ark_stack_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('stack<bool,1>', ark_stack_bool_1, 'ManipulationMsg', 686);

proc ark_stack_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('stack<bool,2>', ark_stack_bool_2, 'ManipulationMsg', 686);

proc ark_stack_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('stack<bool,3>', ark_stack_bool_3, 'ManipulationMsg', 686);

proc ark_stack_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('stack<bool,4>', ark_stack_bool_4, 'ManipulationMsg', 686);

proc ark_stack_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('stack<bigint,1>', ark_stack_bigint_1, 'ManipulationMsg', 686);

proc ark_stack_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('stack<bigint,2>', ark_stack_bigint_2, 'ManipulationMsg', 686);

proc ark_stack_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('stack<bigint,3>', ark_stack_bigint_3, 'ManipulationMsg', 686);

proc ark_stack_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('stack<bigint,4>', ark_stack_bigint_4, 'ManipulationMsg', 686);

proc ark_tile_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('tile<int64,1>', ark_tile_int_1, 'ManipulationMsg', 777);

proc ark_tile_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('tile<int64,2>', ark_tile_int_2, 'ManipulationMsg', 777);

proc ark_tile_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('tile<int64,3>', ark_tile_int_3, 'ManipulationMsg', 777);

proc ark_tile_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('tile<int64,4>', ark_tile_int_4, 'ManipulationMsg', 777);

proc ark_tile_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('tile<uint64,1>', ark_tile_uint_1, 'ManipulationMsg', 777);

proc ark_tile_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('tile<uint64,2>', ark_tile_uint_2, 'ManipulationMsg', 777);

proc ark_tile_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('tile<uint64,3>', ark_tile_uint_3, 'ManipulationMsg', 777);

proc ark_tile_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('tile<uint64,4>', ark_tile_uint_4, 'ManipulationMsg', 777);

proc ark_tile_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('tile<uint8,1>', ark_tile_uint8_1, 'ManipulationMsg', 777);

proc ark_tile_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('tile<uint8,2>', ark_tile_uint8_2, 'ManipulationMsg', 777);

proc ark_tile_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('tile<uint8,3>', ark_tile_uint8_3, 'ManipulationMsg', 777);

proc ark_tile_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('tile<uint8,4>', ark_tile_uint8_4, 'ManipulationMsg', 777);

proc ark_tile_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('tile<float64,1>', ark_tile_real_1, 'ManipulationMsg', 777);

proc ark_tile_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('tile<float64,2>', ark_tile_real_2, 'ManipulationMsg', 777);

proc ark_tile_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('tile<float64,3>', ark_tile_real_3, 'ManipulationMsg', 777);

proc ark_tile_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('tile<float64,4>', ark_tile_real_4, 'ManipulationMsg', 777);

proc ark_tile_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('tile<bool,1>', ark_tile_bool_1, 'ManipulationMsg', 777);

proc ark_tile_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('tile<bool,2>', ark_tile_bool_2, 'ManipulationMsg', 777);

proc ark_tile_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('tile<bool,3>', ark_tile_bool_3, 'ManipulationMsg', 777);

proc ark_tile_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('tile<bool,4>', ark_tile_bool_4, 'ManipulationMsg', 777);

proc ark_tile_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('tile<bigint,1>', ark_tile_bigint_1, 'ManipulationMsg', 777);

proc ark_tile_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('tile<bigint,2>', ark_tile_bigint_2, 'ManipulationMsg', 777);

proc ark_tile_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('tile<bigint,3>', ark_tile_bigint_3, 'ManipulationMsg', 777);

proc ark_tile_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('tile<bigint,4>', ark_tile_bigint_4, 'ManipulationMsg', 777);

proc ark_unstack_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('unstack<int64,1>', ark_unstack_int_1, 'ManipulationMsg', 818);

proc ark_unstack_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('unstack<int64,2>', ark_unstack_int_2, 'ManipulationMsg', 818);

proc ark_unstack_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('unstack<int64,3>', ark_unstack_int_3, 'ManipulationMsg', 818);

proc ark_unstack_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('unstack<int64,4>', ark_unstack_int_4, 'ManipulationMsg', 818);

proc ark_unstack_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('unstack<uint64,1>', ark_unstack_uint_1, 'ManipulationMsg', 818);

proc ark_unstack_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('unstack<uint64,2>', ark_unstack_uint_2, 'ManipulationMsg', 818);

proc ark_unstack_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('unstack<uint64,3>', ark_unstack_uint_3, 'ManipulationMsg', 818);

proc ark_unstack_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('unstack<uint64,4>', ark_unstack_uint_4, 'ManipulationMsg', 818);

proc ark_unstack_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('unstack<uint8,1>', ark_unstack_uint8_1, 'ManipulationMsg', 818);

proc ark_unstack_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('unstack<uint8,2>', ark_unstack_uint8_2, 'ManipulationMsg', 818);

proc ark_unstack_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('unstack<uint8,3>', ark_unstack_uint8_3, 'ManipulationMsg', 818);

proc ark_unstack_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('unstack<uint8,4>', ark_unstack_uint8_4, 'ManipulationMsg', 818);

proc ark_unstack_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('unstack<float64,1>', ark_unstack_real_1, 'ManipulationMsg', 818);

proc ark_unstack_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('unstack<float64,2>', ark_unstack_real_2, 'ManipulationMsg', 818);

proc ark_unstack_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('unstack<float64,3>', ark_unstack_real_3, 'ManipulationMsg', 818);

proc ark_unstack_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('unstack<float64,4>', ark_unstack_real_4, 'ManipulationMsg', 818);

proc ark_unstack_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('unstack<bool,1>', ark_unstack_bool_1, 'ManipulationMsg', 818);

proc ark_unstack_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('unstack<bool,2>', ark_unstack_bool_2, 'ManipulationMsg', 818);

proc ark_unstack_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('unstack<bool,3>', ark_unstack_bool_3, 'ManipulationMsg', 818);

proc ark_unstack_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('unstack<bool,4>', ark_unstack_bool_4, 'ManipulationMsg', 818);

proc ark_unstack_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('unstack<bigint,1>', ark_unstack_bigint_1, 'ManipulationMsg', 818);

proc ark_unstack_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('unstack<bigint,2>', ark_unstack_bigint_2, 'ManipulationMsg', 818);

proc ark_unstack_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('unstack<bigint,3>', ark_unstack_bigint_3, 'ManipulationMsg', 818);

proc ark_unstack_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('unstack<bigint,4>', ark_unstack_bigint_4, 'ManipulationMsg', 818);

proc ark_repeatFlat_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('repeatFlat<int64,1>', ark_repeatFlat_int_1, 'ManipulationMsg', 902);

proc ark_repeatFlat_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('repeatFlat<int64,2>', ark_repeatFlat_int_2, 'ManipulationMsg', 902);

proc ark_repeatFlat_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('repeatFlat<int64,3>', ark_repeatFlat_int_3, 'ManipulationMsg', 902);

proc ark_repeatFlat_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('repeatFlat<int64,4>', ark_repeatFlat_int_4, 'ManipulationMsg', 902);

proc ark_repeatFlat_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('repeatFlat<uint64,1>', ark_repeatFlat_uint_1, 'ManipulationMsg', 902);

proc ark_repeatFlat_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('repeatFlat<uint64,2>', ark_repeatFlat_uint_2, 'ManipulationMsg', 902);

proc ark_repeatFlat_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('repeatFlat<uint64,3>', ark_repeatFlat_uint_3, 'ManipulationMsg', 902);

proc ark_repeatFlat_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('repeatFlat<uint64,4>', ark_repeatFlat_uint_4, 'ManipulationMsg', 902);

proc ark_repeatFlat_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('repeatFlat<uint8,1>', ark_repeatFlat_uint8_1, 'ManipulationMsg', 902);

proc ark_repeatFlat_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('repeatFlat<uint8,2>', ark_repeatFlat_uint8_2, 'ManipulationMsg', 902);

proc ark_repeatFlat_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('repeatFlat<uint8,3>', ark_repeatFlat_uint8_3, 'ManipulationMsg', 902);

proc ark_repeatFlat_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('repeatFlat<uint8,4>', ark_repeatFlat_uint8_4, 'ManipulationMsg', 902);

proc ark_repeatFlat_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('repeatFlat<float64,1>', ark_repeatFlat_real_1, 'ManipulationMsg', 902);

proc ark_repeatFlat_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('repeatFlat<float64,2>', ark_repeatFlat_real_2, 'ManipulationMsg', 902);

proc ark_repeatFlat_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('repeatFlat<float64,3>', ark_repeatFlat_real_3, 'ManipulationMsg', 902);

proc ark_repeatFlat_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('repeatFlat<float64,4>', ark_repeatFlat_real_4, 'ManipulationMsg', 902);

proc ark_repeatFlat_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('repeatFlat<bool,1>', ark_repeatFlat_bool_1, 'ManipulationMsg', 902);

proc ark_repeatFlat_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('repeatFlat<bool,2>', ark_repeatFlat_bool_2, 'ManipulationMsg', 902);

proc ark_repeatFlat_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('repeatFlat<bool,3>', ark_repeatFlat_bool_3, 'ManipulationMsg', 902);

proc ark_repeatFlat_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('repeatFlat<bool,4>', ark_repeatFlat_bool_4, 'ManipulationMsg', 902);

proc ark_repeatFlat_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('repeatFlat<bigint,1>', ark_repeatFlat_bigint_1, 'ManipulationMsg', 902);

proc ark_repeatFlat_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('repeatFlat<bigint,2>', ark_repeatFlat_bigint_2, 'ManipulationMsg', 902);

proc ark_repeatFlat_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('repeatFlat<bigint,3>', ark_repeatFlat_bigint_3, 'ManipulationMsg', 902);

proc ark_repeatFlat_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('repeatFlat<bigint,4>', ark_repeatFlat_bigint_4, 'ManipulationMsg', 902);

import RandMsg;

proc ark_randint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('randint<int64,1>', ark_randint_int_1, 'RandMsg', 36);

proc ark_randint_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('randint<int64,2>', ark_randint_int_2, 'RandMsg', 36);

proc ark_randint_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('randint<int64,3>', ark_randint_int_3, 'RandMsg', 36);

proc ark_randint_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('randint<int64,4>', ark_randint_int_4, 'RandMsg', 36);

proc ark_randint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('randint<uint64,1>', ark_randint_uint_1, 'RandMsg', 36);

proc ark_randint_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('randint<uint64,2>', ark_randint_uint_2, 'RandMsg', 36);

proc ark_randint_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('randint<uint64,3>', ark_randint_uint_3, 'RandMsg', 36);

proc ark_randint_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('randint<uint64,4>', ark_randint_uint_4, 'RandMsg', 36);

proc ark_randint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('randint<uint8,1>', ark_randint_uint8_1, 'RandMsg', 36);

proc ark_randint_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('randint<uint8,2>', ark_randint_uint8_2, 'RandMsg', 36);

proc ark_randint_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('randint<uint8,3>', ark_randint_uint8_3, 'RandMsg', 36);

proc ark_randint_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('randint<uint8,4>', ark_randint_uint8_4, 'RandMsg', 36);

proc ark_randint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('randint<float64,1>', ark_randint_real_1, 'RandMsg', 36);

proc ark_randint_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('randint<float64,2>', ark_randint_real_2, 'RandMsg', 36);

proc ark_randint_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('randint<float64,3>', ark_randint_real_3, 'RandMsg', 36);

proc ark_randint_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('randint<float64,4>', ark_randint_real_4, 'RandMsg', 36);

proc ark_randint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('randint<bool,1>', ark_randint_bool_1, 'RandMsg', 36);

proc ark_randint_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('randint<bool,2>', ark_randint_bool_2, 'RandMsg', 36);

proc ark_randint_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('randint<bool,3>', ark_randint_bool_3, 'RandMsg', 36);

proc ark_randint_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('randint<bool,4>', ark_randint_bool_4, 'RandMsg', 36);

proc ark_randint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('randint<bigint,1>', ark_randint_bigint_1, 'RandMsg', 36);

proc ark_randint_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('randint<bigint,2>', ark_randint_bigint_2, 'RandMsg', 36);

proc ark_randint_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('randint<bigint,3>', ark_randint_bigint_3, 'RandMsg', 36);

proc ark_randint_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randint(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('randint<bigint,4>', ark_randint_bigint_4, 'RandMsg', 36);

proc ark_randomNormal_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randomNormal(cmd, msgArgs, st, array_nd=1);
registerFunction('randomNormal<1>', ark_randomNormal_1, 'RandMsg', 85);

proc ark_randomNormal_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randomNormal(cmd, msgArgs, st, array_nd=2);
registerFunction('randomNormal<2>', ark_randomNormal_2, 'RandMsg', 85);

proc ark_randomNormal_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randomNormal(cmd, msgArgs, st, array_nd=3);
registerFunction('randomNormal<3>', ark_randomNormal_3, 'RandMsg', 85);

proc ark_randomNormal_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.randomNormal(cmd, msgArgs, st, array_nd=4);
registerFunction('randomNormal<4>', ark_randomNormal_4, 'RandMsg', 85);

proc ark_createGenerator_int(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.createGenerator(cmd, msgArgs, st, array_dtype=int);
registerFunction('createGenerator<int64>', ark_createGenerator_int, 'RandMsg', 99);

proc ark_createGenerator_uint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.createGenerator(cmd, msgArgs, st, array_dtype=uint);
registerFunction('createGenerator<uint64>', ark_createGenerator_uint, 'RandMsg', 99);

proc ark_createGenerator_uint8(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.createGenerator(cmd, msgArgs, st, array_dtype=uint(8));
registerFunction('createGenerator<uint8>', ark_createGenerator_uint8, 'RandMsg', 99);

proc ark_createGenerator_real(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.createGenerator(cmd, msgArgs, st, array_dtype=real);
registerFunction('createGenerator<float64>', ark_createGenerator_real, 'RandMsg', 99);

proc ark_createGenerator_bool(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.createGenerator(cmd, msgArgs, st, array_dtype=bool);
registerFunction('createGenerator<bool>', ark_createGenerator_bool, 'RandMsg', 99);

proc ark_createGenerator_bigint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.createGenerator(cmd, msgArgs, st, array_dtype=bigint);
registerFunction('createGenerator<bigint>', ark_createGenerator_bigint, 'RandMsg', 99);

proc ark_uniformGenerator_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('uniformGenerator<int64,1>', ark_uniformGenerator_int_1, 'RandMsg', 127);

proc ark_uniformGenerator_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('uniformGenerator<int64,2>', ark_uniformGenerator_int_2, 'RandMsg', 127);

proc ark_uniformGenerator_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('uniformGenerator<int64,3>', ark_uniformGenerator_int_3, 'RandMsg', 127);

proc ark_uniformGenerator_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('uniformGenerator<int64,4>', ark_uniformGenerator_int_4, 'RandMsg', 127);

proc ark_uniformGenerator_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('uniformGenerator<uint64,1>', ark_uniformGenerator_uint_1, 'RandMsg', 127);

proc ark_uniformGenerator_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('uniformGenerator<uint64,2>', ark_uniformGenerator_uint_2, 'RandMsg', 127);

proc ark_uniformGenerator_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('uniformGenerator<uint64,3>', ark_uniformGenerator_uint_3, 'RandMsg', 127);

proc ark_uniformGenerator_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('uniformGenerator<uint64,4>', ark_uniformGenerator_uint_4, 'RandMsg', 127);

proc ark_uniformGenerator_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('uniformGenerator<uint8,1>', ark_uniformGenerator_uint8_1, 'RandMsg', 127);

proc ark_uniformGenerator_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('uniformGenerator<uint8,2>', ark_uniformGenerator_uint8_2, 'RandMsg', 127);

proc ark_uniformGenerator_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('uniformGenerator<uint8,3>', ark_uniformGenerator_uint8_3, 'RandMsg', 127);

proc ark_uniformGenerator_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('uniformGenerator<uint8,4>', ark_uniformGenerator_uint8_4, 'RandMsg', 127);

proc ark_uniformGenerator_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('uniformGenerator<float64,1>', ark_uniformGenerator_real_1, 'RandMsg', 127);

proc ark_uniformGenerator_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('uniformGenerator<float64,2>', ark_uniformGenerator_real_2, 'RandMsg', 127);

proc ark_uniformGenerator_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('uniformGenerator<float64,3>', ark_uniformGenerator_real_3, 'RandMsg', 127);

proc ark_uniformGenerator_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('uniformGenerator<float64,4>', ark_uniformGenerator_real_4, 'RandMsg', 127);

proc ark_uniformGenerator_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('uniformGenerator<bool,1>', ark_uniformGenerator_bool_1, 'RandMsg', 127);

proc ark_uniformGenerator_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('uniformGenerator<bool,2>', ark_uniformGenerator_bool_2, 'RandMsg', 127);

proc ark_uniformGenerator_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('uniformGenerator<bool,3>', ark_uniformGenerator_bool_3, 'RandMsg', 127);

proc ark_uniformGenerator_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('uniformGenerator<bool,4>', ark_uniformGenerator_bool_4, 'RandMsg', 127);

proc ark_uniformGenerator_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('uniformGenerator<bigint,1>', ark_uniformGenerator_bigint_1, 'RandMsg', 127);

proc ark_uniformGenerator_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('uniformGenerator<bigint,2>', ark_uniformGenerator_bigint_2, 'RandMsg', 127);

proc ark_uniformGenerator_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('uniformGenerator<bigint,3>', ark_uniformGenerator_bigint_3, 'RandMsg', 127);

proc ark_uniformGenerator_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.uniformGenerator(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('uniformGenerator<bigint,4>', ark_uniformGenerator_bigint_4, 'RandMsg', 127);

proc ark_standardNormalGenerator_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.standardNormalGenerator(cmd, msgArgs, st, array_nd=1);
registerFunction('standardNormalGenerator<1>', ark_standardNormalGenerator_1, 'RandMsg', 254);

proc ark_standardExponential_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.standardExponential(cmd, msgArgs, st, array_nd=1);
registerFunction('standardExponential<1>', ark_standardExponential_1, 'RandMsg', 389);

proc ark_standardNormalGenerator_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.standardNormalGenerator(cmd, msgArgs, st, array_nd=2);
registerFunction('standardNormalGenerator<2>', ark_standardNormalGenerator_2, 'RandMsg', 161);

proc ark_standardNormalGenerator_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.standardNormalGenerator(cmd, msgArgs, st, array_nd=3);
registerFunction('standardNormalGenerator<3>', ark_standardNormalGenerator_3, 'RandMsg', 161);

proc ark_standardNormalGenerator_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.standardNormalGenerator(cmd, msgArgs, st, array_nd=4);
registerFunction('standardNormalGenerator<4>', ark_standardNormalGenerator_4, 'RandMsg', 161);

proc ark_choice_int(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.choice(cmd, msgArgs, st, array_dtype=int);
registerFunction('choice<int64>', ark_choice_int, 'RandMsg', 503);

proc ark_choice_uint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.choice(cmd, msgArgs, st, array_dtype=uint);
registerFunction('choice<uint64>', ark_choice_uint, 'RandMsg', 503);

proc ark_choice_uint8(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.choice(cmd, msgArgs, st, array_dtype=uint(8));
registerFunction('choice<uint8>', ark_choice_uint8, 'RandMsg', 503);

proc ark_choice_real(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.choice(cmd, msgArgs, st, array_dtype=real);
registerFunction('choice<float64>', ark_choice_real, 'RandMsg', 503);

proc ark_choice_bool(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.choice(cmd, msgArgs, st, array_dtype=bool);
registerFunction('choice<bool>', ark_choice_bool, 'RandMsg', 503);

proc ark_choice_bigint(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.choice(cmd, msgArgs, st, array_dtype=bigint);
registerFunction('choice<bigint>', ark_choice_bigint, 'RandMsg', 503);

proc ark_permutation_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('permutation<int64,1>', ark_permutation_int_1, 'RandMsg', 577);

proc ark_permutation_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('permutation<int64,2>', ark_permutation_int_2, 'RandMsg', 406);

proc ark_permutation_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('permutation<int64,3>', ark_permutation_int_3, 'RandMsg', 406);

proc ark_permutation_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('permutation<int64,4>', ark_permutation_int_4, 'RandMsg', 406);

proc ark_permutation_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('permutation<uint64,1>', ark_permutation_uint_1, 'RandMsg', 577);

proc ark_permutation_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('permutation<uint64,2>', ark_permutation_uint_2, 'RandMsg', 406);

proc ark_permutation_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('permutation<uint64,3>', ark_permutation_uint_3, 'RandMsg', 406);

proc ark_permutation_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('permutation<uint64,4>', ark_permutation_uint_4, 'RandMsg', 406);

proc ark_permutation_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('permutation<uint8,1>', ark_permutation_uint8_1, 'RandMsg', 577);

proc ark_permutation_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('permutation<uint8,2>', ark_permutation_uint8_2, 'RandMsg', 406);

proc ark_permutation_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('permutation<uint8,3>', ark_permutation_uint8_3, 'RandMsg', 406);

proc ark_permutation_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('permutation<uint8,4>', ark_permutation_uint8_4, 'RandMsg', 406);

proc ark_permutation_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('permutation<float64,1>', ark_permutation_real_1, 'RandMsg', 577);

proc ark_permutation_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('permutation<float64,2>', ark_permutation_real_2, 'RandMsg', 406);

proc ark_permutation_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('permutation<float64,3>', ark_permutation_real_3, 'RandMsg', 406);

proc ark_permutation_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('permutation<float64,4>', ark_permutation_real_4, 'RandMsg', 406);

proc ark_permutation_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('permutation<bool,1>', ark_permutation_bool_1, 'RandMsg', 577);

proc ark_permutation_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('permutation<bool,2>', ark_permutation_bool_2, 'RandMsg', 406);

proc ark_permutation_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('permutation<bool,3>', ark_permutation_bool_3, 'RandMsg', 406);

proc ark_permutation_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('permutation<bool,4>', ark_permutation_bool_4, 'RandMsg', 406);

proc ark_permutation_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('permutation<bigint,1>', ark_permutation_bigint_1, 'RandMsg', 577);

proc ark_permutation_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('permutation<bigint,2>', ark_permutation_bigint_2, 'RandMsg', 406);

proc ark_permutation_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('permutation<bigint,3>', ark_permutation_bigint_3, 'RandMsg', 406);

proc ark_permutation_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.permutation(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('permutation<bigint,4>', ark_permutation_bigint_4, 'RandMsg', 406);

proc ark_shuffle_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('shuffle<int64,1>', ark_shuffle_int_1, 'RandMsg', 664);

proc ark_shuffle_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('shuffle<int64,2>', ark_shuffle_int_2, 'RandMsg', 493);

proc ark_shuffle_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('shuffle<int64,3>', ark_shuffle_int_3, 'RandMsg', 493);

proc ark_shuffle_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('shuffle<int64,4>', ark_shuffle_int_4, 'RandMsg', 493);

proc ark_shuffle_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('shuffle<uint64,1>', ark_shuffle_uint_1, 'RandMsg', 664);

proc ark_shuffle_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('shuffle<uint64,2>', ark_shuffle_uint_2, 'RandMsg', 493);

proc ark_shuffle_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('shuffle<uint64,3>', ark_shuffle_uint_3, 'RandMsg', 493);

proc ark_shuffle_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('shuffle<uint64,4>', ark_shuffle_uint_4, 'RandMsg', 493);

proc ark_shuffle_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('shuffle<uint8,1>', ark_shuffle_uint8_1, 'RandMsg', 664);

proc ark_shuffle_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('shuffle<uint8,2>', ark_shuffle_uint8_2, 'RandMsg', 493);

proc ark_shuffle_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('shuffle<uint8,3>', ark_shuffle_uint8_3, 'RandMsg', 493);

proc ark_shuffle_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('shuffle<uint8,4>', ark_shuffle_uint8_4, 'RandMsg', 493);

proc ark_shuffle_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('shuffle<float64,1>', ark_shuffle_real_1, 'RandMsg', 664);

proc ark_shuffle_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('shuffle<float64,2>', ark_shuffle_real_2, 'RandMsg', 493);

proc ark_shuffle_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('shuffle<float64,3>', ark_shuffle_real_3, 'RandMsg', 493);

proc ark_shuffle_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('shuffle<float64,4>', ark_shuffle_real_4, 'RandMsg', 493);

proc ark_shuffle_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('shuffle<bool,1>', ark_shuffle_bool_1, 'RandMsg', 664);

proc ark_shuffle_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('shuffle<bool,2>', ark_shuffle_bool_2, 'RandMsg', 493);

proc ark_shuffle_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('shuffle<bool,3>', ark_shuffle_bool_3, 'RandMsg', 493);

proc ark_shuffle_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('shuffle<bool,4>', ark_shuffle_bool_4, 'RandMsg', 493);

proc ark_shuffle_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('shuffle<bigint,1>', ark_shuffle_bigint_1, 'RandMsg', 664);

proc ark_shuffle_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('shuffle<bigint,2>', ark_shuffle_bigint_2, 'RandMsg', 493);

proc ark_shuffle_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('shuffle<bigint,3>', ark_shuffle_bigint_3, 'RandMsg', 493);

proc ark_shuffle_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return RandMsg.shuffle(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('shuffle<bigint,4>', ark_shuffle_bigint_4, 'RandMsg', 493);

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

proc ark_mean_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2);
registerFunction('mean<int64,2>', ark_mean_int_2, 'StatsMsg', 22);

proc ark_mean_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3);
registerFunction('mean<int64,3>', ark_mean_int_3, 'StatsMsg', 22);

proc ark_mean_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4);
registerFunction('mean<int64,4>', ark_mean_int_4, 'StatsMsg', 22);

proc ark_mean_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('mean<uint64,1>', ark_mean_uint_1, 'StatsMsg', 22);

proc ark_mean_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2);
registerFunction('mean<uint64,2>', ark_mean_uint_2, 'StatsMsg', 22);

proc ark_mean_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3);
registerFunction('mean<uint64,3>', ark_mean_uint_3, 'StatsMsg', 22);

proc ark_mean_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4);
registerFunction('mean<uint64,4>', ark_mean_uint_4, 'StatsMsg', 22);

proc ark_mean_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('mean<uint8,1>', ark_mean_uint8_1, 'StatsMsg', 22);

proc ark_mean_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2);
registerFunction('mean<uint8,2>', ark_mean_uint8_2, 'StatsMsg', 22);

proc ark_mean_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3);
registerFunction('mean<uint8,3>', ark_mean_uint8_3, 'StatsMsg', 22);

proc ark_mean_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4);
registerFunction('mean<uint8,4>', ark_mean_uint8_4, 'StatsMsg', 22);

proc ark_mean_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('mean<float64,1>', ark_mean_real_1, 'StatsMsg', 22);

proc ark_mean_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2);
registerFunction('mean<float64,2>', ark_mean_real_2, 'StatsMsg', 22);

proc ark_mean_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3);
registerFunction('mean<float64,3>', ark_mean_real_3, 'StatsMsg', 22);

proc ark_mean_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4);
registerFunction('mean<float64,4>', ark_mean_real_4, 'StatsMsg', 22);

proc ark_mean_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('mean<bool,1>', ark_mean_bool_1, 'StatsMsg', 22);

proc ark_mean_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2);
registerFunction('mean<bool,2>', ark_mean_bool_2, 'StatsMsg', 22);

proc ark_mean_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3);
registerFunction('mean<bool,3>', ark_mean_bool_3, 'StatsMsg', 22);

proc ark_mean_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4);
registerFunction('mean<bool,4>', ark_mean_bool_4, 'StatsMsg', 22);

proc ark_mean_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('mean<bigint,1>', ark_mean_bigint_1, 'StatsMsg', 22);

proc ark_mean_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2);
registerFunction('mean<bigint,2>', ark_mean_bigint_2, 'StatsMsg', 22);

proc ark_mean_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3);
registerFunction('mean<bigint,3>', ark_mean_bigint_3, 'StatsMsg', 22);

proc ark_mean_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4);
registerFunction('mean<bigint,4>', ark_mean_bigint_4, 'StatsMsg', 22);

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

proc ark_meanReduce_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2);
registerFunction('meanReduce<int64,2>', ark_meanReduce_int_2, 'StatsMsg', 29);

proc ark_meanReduce_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3);
registerFunction('meanReduce<int64,3>', ark_meanReduce_int_3, 'StatsMsg', 29);

proc ark_meanReduce_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4);
registerFunction('meanReduce<int64,4>', ark_meanReduce_int_4, 'StatsMsg', 29);

proc ark_meanReduce_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('meanReduce<uint64,1>', ark_meanReduce_uint_1, 'StatsMsg', 29);

proc ark_meanReduce_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2);
registerFunction('meanReduce<uint64,2>', ark_meanReduce_uint_2, 'StatsMsg', 29);

proc ark_meanReduce_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3);
registerFunction('meanReduce<uint64,3>', ark_meanReduce_uint_3, 'StatsMsg', 29);

proc ark_meanReduce_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4);
registerFunction('meanReduce<uint64,4>', ark_meanReduce_uint_4, 'StatsMsg', 29);

proc ark_meanReduce_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('meanReduce<uint8,1>', ark_meanReduce_uint8_1, 'StatsMsg', 29);

proc ark_meanReduce_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2);
registerFunction('meanReduce<uint8,2>', ark_meanReduce_uint8_2, 'StatsMsg', 29);

proc ark_meanReduce_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3);
registerFunction('meanReduce<uint8,3>', ark_meanReduce_uint8_3, 'StatsMsg', 29);

proc ark_meanReduce_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4);
registerFunction('meanReduce<uint8,4>', ark_meanReduce_uint8_4, 'StatsMsg', 29);

proc ark_meanReduce_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('meanReduce<float64,1>', ark_meanReduce_real_1, 'StatsMsg', 29);

proc ark_meanReduce_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2);
registerFunction('meanReduce<float64,2>', ark_meanReduce_real_2, 'StatsMsg', 29);

proc ark_meanReduce_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3);
registerFunction('meanReduce<float64,3>', ark_meanReduce_real_3, 'StatsMsg', 29);

proc ark_meanReduce_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4);
registerFunction('meanReduce<float64,4>', ark_meanReduce_real_4, 'StatsMsg', 29);

proc ark_meanReduce_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('meanReduce<bool,1>', ark_meanReduce_bool_1, 'StatsMsg', 29);

proc ark_meanReduce_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2);
registerFunction('meanReduce<bool,2>', ark_meanReduce_bool_2, 'StatsMsg', 29);

proc ark_meanReduce_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3);
registerFunction('meanReduce<bool,3>', ark_meanReduce_bool_3, 'StatsMsg', 29);

proc ark_meanReduce_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4);
registerFunction('meanReduce<bool,4>', ark_meanReduce_bool_4, 'StatsMsg', 29);

proc ark_meanReduce_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('meanReduce<bigint,1>', ark_meanReduce_bigint_1, 'StatsMsg', 29);

proc ark_meanReduce_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2);
registerFunction('meanReduce<bigint,2>', ark_meanReduce_bigint_2, 'StatsMsg', 29);

proc ark_meanReduce_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3);
registerFunction('meanReduce<bigint,3>', ark_meanReduce_bigint_3, 'StatsMsg', 29);

proc ark_meanReduce_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4);
registerFunction('meanReduce<bigint,4>', ark_meanReduce_bigint_4, 'StatsMsg', 29);

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

proc ark_var_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2);
registerFunction('var<int64,2>', ark_var_int_2, 'StatsMsg', 40);

proc ark_var_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3);
registerFunction('var<int64,3>', ark_var_int_3, 'StatsMsg', 40);

proc ark_var_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4);
registerFunction('var<int64,4>', ark_var_int_4, 'StatsMsg', 40);

proc ark_var_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('var<uint64,1>', ark_var_uint_1, 'StatsMsg', 40);

proc ark_var_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2);
registerFunction('var<uint64,2>', ark_var_uint_2, 'StatsMsg', 40);

proc ark_var_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3);
registerFunction('var<uint64,3>', ark_var_uint_3, 'StatsMsg', 40);

proc ark_var_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4);
registerFunction('var<uint64,4>', ark_var_uint_4, 'StatsMsg', 40);

proc ark_var_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('var<uint8,1>', ark_var_uint8_1, 'StatsMsg', 40);

proc ark_var_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2);
registerFunction('var<uint8,2>', ark_var_uint8_2, 'StatsMsg', 40);

proc ark_var_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3);
registerFunction('var<uint8,3>', ark_var_uint8_3, 'StatsMsg', 40);

proc ark_var_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4);
registerFunction('var<uint8,4>', ark_var_uint8_4, 'StatsMsg', 40);

proc ark_var_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('var<float64,1>', ark_var_real_1, 'StatsMsg', 40);

proc ark_var_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2);
registerFunction('var<float64,2>', ark_var_real_2, 'StatsMsg', 40);

proc ark_var_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3);
registerFunction('var<float64,3>', ark_var_real_3, 'StatsMsg', 40);

proc ark_var_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4);
registerFunction('var<float64,4>', ark_var_real_4, 'StatsMsg', 40);

proc ark_var_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('var<bool,1>', ark_var_bool_1, 'StatsMsg', 40);

proc ark_var_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2);
registerFunction('var<bool,2>', ark_var_bool_2, 'StatsMsg', 40);

proc ark_var_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3);
registerFunction('var<bool,3>', ark_var_bool_3, 'StatsMsg', 40);

proc ark_var_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4);
registerFunction('var<bool,4>', ark_var_bool_4, 'StatsMsg', 40);

proc ark_var_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('var<bigint,1>', ark_var_bigint_1, 'StatsMsg', 40);

proc ark_var_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2);
registerFunction('var<bigint,2>', ark_var_bigint_2, 'StatsMsg', 40);

proc ark_var_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3);
registerFunction('var<bigint,3>', ark_var_bigint_3, 'StatsMsg', 40);

proc ark_var_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_variance_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4);
registerFunction('var<bigint,4>', ark_var_bigint_4, 'StatsMsg', 40);

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

proc ark_varReduce_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2);
registerFunction('varReduce<int64,2>', ark_varReduce_int_2, 'StatsMsg', 47);

proc ark_varReduce_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3);
registerFunction('varReduce<int64,3>', ark_varReduce_int_3, 'StatsMsg', 47);

proc ark_varReduce_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4);
registerFunction('varReduce<int64,4>', ark_varReduce_int_4, 'StatsMsg', 47);

proc ark_varReduce_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('varReduce<uint64,1>', ark_varReduce_uint_1, 'StatsMsg', 47);

proc ark_varReduce_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2);
registerFunction('varReduce<uint64,2>', ark_varReduce_uint_2, 'StatsMsg', 47);

proc ark_varReduce_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3);
registerFunction('varReduce<uint64,3>', ark_varReduce_uint_3, 'StatsMsg', 47);

proc ark_varReduce_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4);
registerFunction('varReduce<uint64,4>', ark_varReduce_uint_4, 'StatsMsg', 47);

proc ark_varReduce_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('varReduce<uint8,1>', ark_varReduce_uint8_1, 'StatsMsg', 47);

proc ark_varReduce_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2);
registerFunction('varReduce<uint8,2>', ark_varReduce_uint8_2, 'StatsMsg', 47);

proc ark_varReduce_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3);
registerFunction('varReduce<uint8,3>', ark_varReduce_uint8_3, 'StatsMsg', 47);

proc ark_varReduce_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4);
registerFunction('varReduce<uint8,4>', ark_varReduce_uint8_4, 'StatsMsg', 47);

proc ark_varReduce_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('varReduce<float64,1>', ark_varReduce_real_1, 'StatsMsg', 47);

proc ark_varReduce_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2);
registerFunction('varReduce<float64,2>', ark_varReduce_real_2, 'StatsMsg', 47);

proc ark_varReduce_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3);
registerFunction('varReduce<float64,3>', ark_varReduce_real_3, 'StatsMsg', 47);

proc ark_varReduce_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4);
registerFunction('varReduce<float64,4>', ark_varReduce_real_4, 'StatsMsg', 47);

proc ark_varReduce_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('varReduce<bool,1>', ark_varReduce_bool_1, 'StatsMsg', 47);

proc ark_varReduce_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2);
registerFunction('varReduce<bool,2>', ark_varReduce_bool_2, 'StatsMsg', 47);

proc ark_varReduce_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3);
registerFunction('varReduce<bool,3>', ark_varReduce_bool_3, 'StatsMsg', 47);

proc ark_varReduce_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4);
registerFunction('varReduce<bool,4>', ark_varReduce_bool_4, 'StatsMsg', 47);

proc ark_varReduce_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('varReduce<bigint,1>', ark_varReduce_bigint_1, 'StatsMsg', 47);

proc ark_varReduce_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2);
registerFunction('varReduce<bigint,2>', ark_varReduce_bigint_2, 'StatsMsg', 47);

proc ark_varReduce_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3);
registerFunction('varReduce<bigint,3>', ark_varReduce_bigint_3, 'StatsMsg', 47);

proc ark_varReduce_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_varReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4);
registerFunction('varReduce<bigint,4>', ark_varReduce_bigint_4, 'StatsMsg', 47);

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

proc ark_std_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2);
registerFunction('std<int64,2>', ark_std_int_2, 'StatsMsg', 58);

proc ark_std_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3);
registerFunction('std<int64,3>', ark_std_int_3, 'StatsMsg', 58);

proc ark_std_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4);
registerFunction('std<int64,4>', ark_std_int_4, 'StatsMsg', 58);

proc ark_std_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('std<uint64,1>', ark_std_uint_1, 'StatsMsg', 58);

proc ark_std_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2);
registerFunction('std<uint64,2>', ark_std_uint_2, 'StatsMsg', 58);

proc ark_std_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3);
registerFunction('std<uint64,3>', ark_std_uint_3, 'StatsMsg', 58);

proc ark_std_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4);
registerFunction('std<uint64,4>', ark_std_uint_4, 'StatsMsg', 58);

proc ark_std_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('std<uint8,1>', ark_std_uint8_1, 'StatsMsg', 58);

proc ark_std_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2);
registerFunction('std<uint8,2>', ark_std_uint8_2, 'StatsMsg', 58);

proc ark_std_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3);
registerFunction('std<uint8,3>', ark_std_uint8_3, 'StatsMsg', 58);

proc ark_std_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4);
registerFunction('std<uint8,4>', ark_std_uint8_4, 'StatsMsg', 58);

proc ark_std_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('std<float64,1>', ark_std_real_1, 'StatsMsg', 58);

proc ark_std_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2);
registerFunction('std<float64,2>', ark_std_real_2, 'StatsMsg', 58);

proc ark_std_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3);
registerFunction('std<float64,3>', ark_std_real_3, 'StatsMsg', 58);

proc ark_std_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4);
registerFunction('std<float64,4>', ark_std_real_4, 'StatsMsg', 58);

proc ark_std_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('std<bool,1>', ark_std_bool_1, 'StatsMsg', 58);

proc ark_std_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2);
registerFunction('std<bool,2>', ark_std_bool_2, 'StatsMsg', 58);

proc ark_std_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3);
registerFunction('std<bool,3>', ark_std_bool_3, 'StatsMsg', 58);

proc ark_std_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4);
registerFunction('std<bool,4>', ark_std_bool_4, 'StatsMsg', 58);

proc ark_std_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('std<bigint,1>', ark_std_bigint_1, 'StatsMsg', 58);

proc ark_std_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2);
registerFunction('std<bigint,2>', ark_std_bigint_2, 'StatsMsg', 58);

proc ark_std_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3);
registerFunction('std<bigint,3>', ark_std_bigint_3, 'StatsMsg', 58);

proc ark_std_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_std_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4);
registerFunction('std<bigint,4>', ark_std_bigint_4, 'StatsMsg', 58);

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

proc ark_stdReduce_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2);
registerFunction('stdReduce<int64,2>', ark_stdReduce_int_2, 'StatsMsg', 65);

proc ark_stdReduce_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3);
registerFunction('stdReduce<int64,3>', ark_stdReduce_int_3, 'StatsMsg', 65);

proc ark_stdReduce_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4);
registerFunction('stdReduce<int64,4>', ark_stdReduce_int_4, 'StatsMsg', 65);

proc ark_stdReduce_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('stdReduce<uint64,1>', ark_stdReduce_uint_1, 'StatsMsg', 65);

proc ark_stdReduce_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2);
registerFunction('stdReduce<uint64,2>', ark_stdReduce_uint_2, 'StatsMsg', 65);

proc ark_stdReduce_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3);
registerFunction('stdReduce<uint64,3>', ark_stdReduce_uint_3, 'StatsMsg', 65);

proc ark_stdReduce_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4);
registerFunction('stdReduce<uint64,4>', ark_stdReduce_uint_4, 'StatsMsg', 65);

proc ark_stdReduce_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('stdReduce<uint8,1>', ark_stdReduce_uint8_1, 'StatsMsg', 65);

proc ark_stdReduce_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2);
registerFunction('stdReduce<uint8,2>', ark_stdReduce_uint8_2, 'StatsMsg', 65);

proc ark_stdReduce_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3);
registerFunction('stdReduce<uint8,3>', ark_stdReduce_uint8_3, 'StatsMsg', 65);

proc ark_stdReduce_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4);
registerFunction('stdReduce<uint8,4>', ark_stdReduce_uint8_4, 'StatsMsg', 65);

proc ark_stdReduce_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('stdReduce<float64,1>', ark_stdReduce_real_1, 'StatsMsg', 65);

proc ark_stdReduce_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2);
registerFunction('stdReduce<float64,2>', ark_stdReduce_real_2, 'StatsMsg', 65);

proc ark_stdReduce_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3);
registerFunction('stdReduce<float64,3>', ark_stdReduce_real_3, 'StatsMsg', 65);

proc ark_stdReduce_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4);
registerFunction('stdReduce<float64,4>', ark_stdReduce_real_4, 'StatsMsg', 65);

proc ark_stdReduce_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('stdReduce<bool,1>', ark_stdReduce_bool_1, 'StatsMsg', 65);

proc ark_stdReduce_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2);
registerFunction('stdReduce<bool,2>', ark_stdReduce_bool_2, 'StatsMsg', 65);

proc ark_stdReduce_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3);
registerFunction('stdReduce<bool,3>', ark_stdReduce_bool_3, 'StatsMsg', 65);

proc ark_stdReduce_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4);
registerFunction('stdReduce<bool,4>', ark_stdReduce_bool_4, 'StatsMsg', 65);

proc ark_stdReduce_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('stdReduce<bigint,1>', ark_stdReduce_bigint_1, 'StatsMsg', 65);

proc ark_stdReduce_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2);
registerFunction('stdReduce<bigint,2>', ark_stdReduce_bigint_2, 'StatsMsg', 65);

proc ark_stdReduce_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3);
registerFunction('stdReduce<bigint,3>', ark_stdReduce_bigint_3, 'StatsMsg', 65);

proc ark_stdReduce_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_stdReduce_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4);
registerFunction('stdReduce<bigint,4>', ark_stdReduce_bigint_4, 'StatsMsg', 65);

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

proc ark_cov_int_1_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<int64,1,int64,2>', ark_cov_int_1_int_2, 'StatsMsg', 76);

proc ark_cov_int_1_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<int64,1,int64,3>', ark_cov_int_1_int_3, 'StatsMsg', 76);

proc ark_cov_int_1_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<int64,1,int64,4>', ark_cov_int_1_int_4, 'StatsMsg', 76);

proc ark_cov_int_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<int64,1,uint64,1>', ark_cov_int_1_uint_1, 'StatsMsg', 76);

proc ark_cov_int_1_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<int64,1,uint64,2>', ark_cov_int_1_uint_2, 'StatsMsg', 76);

proc ark_cov_int_1_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<int64,1,uint64,3>', ark_cov_int_1_uint_3, 'StatsMsg', 76);

proc ark_cov_int_1_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<int64,1,uint64,4>', ark_cov_int_1_uint_4, 'StatsMsg', 76);

proc ark_cov_int_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<int64,1,uint8,1>', ark_cov_int_1_uint8_1, 'StatsMsg', 76);

proc ark_cov_int_1_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<int64,1,uint8,2>', ark_cov_int_1_uint8_2, 'StatsMsg', 76);

proc ark_cov_int_1_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<int64,1,uint8,3>', ark_cov_int_1_uint8_3, 'StatsMsg', 76);

proc ark_cov_int_1_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<int64,1,uint8,4>', ark_cov_int_1_uint8_4, 'StatsMsg', 76);

proc ark_cov_int_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<int64,1,float64,1>', ark_cov_int_1_real_1, 'StatsMsg', 76);

proc ark_cov_int_1_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<int64,1,float64,2>', ark_cov_int_1_real_2, 'StatsMsg', 76);

proc ark_cov_int_1_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<int64,1,float64,3>', ark_cov_int_1_real_3, 'StatsMsg', 76);

proc ark_cov_int_1_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<int64,1,float64,4>', ark_cov_int_1_real_4, 'StatsMsg', 76);

proc ark_cov_int_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<int64,1,bool,1>', ark_cov_int_1_bool_1, 'StatsMsg', 76);

proc ark_cov_int_1_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<int64,1,bool,2>', ark_cov_int_1_bool_2, 'StatsMsg', 76);

proc ark_cov_int_1_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<int64,1,bool,3>', ark_cov_int_1_bool_3, 'StatsMsg', 76);

proc ark_cov_int_1_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<int64,1,bool,4>', ark_cov_int_1_bool_4, 'StatsMsg', 76);

proc ark_cov_int_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<int64,1,bigint,1>', ark_cov_int_1_bigint_1, 'StatsMsg', 76);

proc ark_cov_int_1_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<int64,1,bigint,2>', ark_cov_int_1_bigint_2, 'StatsMsg', 76);

proc ark_cov_int_1_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<int64,1,bigint,3>', ark_cov_int_1_bigint_3, 'StatsMsg', 76);

proc ark_cov_int_1_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<int64,1,bigint,4>', ark_cov_int_1_bigint_4, 'StatsMsg', 76);

proc ark_cov_int_2_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<int64,2,int64,1>', ark_cov_int_2_int_1, 'StatsMsg', 76);

proc ark_cov_int_2_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<int64,2,int64,2>', ark_cov_int_2_int_2, 'StatsMsg', 76);

proc ark_cov_int_2_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<int64,2,int64,3>', ark_cov_int_2_int_3, 'StatsMsg', 76);

proc ark_cov_int_2_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<int64,2,int64,4>', ark_cov_int_2_int_4, 'StatsMsg', 76);

proc ark_cov_int_2_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<int64,2,uint64,1>', ark_cov_int_2_uint_1, 'StatsMsg', 76);

proc ark_cov_int_2_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<int64,2,uint64,2>', ark_cov_int_2_uint_2, 'StatsMsg', 76);

proc ark_cov_int_2_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<int64,2,uint64,3>', ark_cov_int_2_uint_3, 'StatsMsg', 76);

proc ark_cov_int_2_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<int64,2,uint64,4>', ark_cov_int_2_uint_4, 'StatsMsg', 76);

proc ark_cov_int_2_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<int64,2,uint8,1>', ark_cov_int_2_uint8_1, 'StatsMsg', 76);

proc ark_cov_int_2_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<int64,2,uint8,2>', ark_cov_int_2_uint8_2, 'StatsMsg', 76);

proc ark_cov_int_2_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<int64,2,uint8,3>', ark_cov_int_2_uint8_3, 'StatsMsg', 76);

proc ark_cov_int_2_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<int64,2,uint8,4>', ark_cov_int_2_uint8_4, 'StatsMsg', 76);

proc ark_cov_int_2_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<int64,2,float64,1>', ark_cov_int_2_real_1, 'StatsMsg', 76);

proc ark_cov_int_2_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<int64,2,float64,2>', ark_cov_int_2_real_2, 'StatsMsg', 76);

proc ark_cov_int_2_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<int64,2,float64,3>', ark_cov_int_2_real_3, 'StatsMsg', 76);

proc ark_cov_int_2_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<int64,2,float64,4>', ark_cov_int_2_real_4, 'StatsMsg', 76);

proc ark_cov_int_2_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<int64,2,bool,1>', ark_cov_int_2_bool_1, 'StatsMsg', 76);

proc ark_cov_int_2_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<int64,2,bool,2>', ark_cov_int_2_bool_2, 'StatsMsg', 76);

proc ark_cov_int_2_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<int64,2,bool,3>', ark_cov_int_2_bool_3, 'StatsMsg', 76);

proc ark_cov_int_2_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<int64,2,bool,4>', ark_cov_int_2_bool_4, 'StatsMsg', 76);

proc ark_cov_int_2_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<int64,2,bigint,1>', ark_cov_int_2_bigint_1, 'StatsMsg', 76);

proc ark_cov_int_2_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<int64,2,bigint,2>', ark_cov_int_2_bigint_2, 'StatsMsg', 76);

proc ark_cov_int_2_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<int64,2,bigint,3>', ark_cov_int_2_bigint_3, 'StatsMsg', 76);

proc ark_cov_int_2_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<int64,2,bigint,4>', ark_cov_int_2_bigint_4, 'StatsMsg', 76);

proc ark_cov_int_3_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<int64,3,int64,1>', ark_cov_int_3_int_1, 'StatsMsg', 76);

proc ark_cov_int_3_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<int64,3,int64,2>', ark_cov_int_3_int_2, 'StatsMsg', 76);

proc ark_cov_int_3_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<int64,3,int64,3>', ark_cov_int_3_int_3, 'StatsMsg', 76);

proc ark_cov_int_3_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<int64,3,int64,4>', ark_cov_int_3_int_4, 'StatsMsg', 76);

proc ark_cov_int_3_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<int64,3,uint64,1>', ark_cov_int_3_uint_1, 'StatsMsg', 76);

proc ark_cov_int_3_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<int64,3,uint64,2>', ark_cov_int_3_uint_2, 'StatsMsg', 76);

proc ark_cov_int_3_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<int64,3,uint64,3>', ark_cov_int_3_uint_3, 'StatsMsg', 76);

proc ark_cov_int_3_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<int64,3,uint64,4>', ark_cov_int_3_uint_4, 'StatsMsg', 76);

proc ark_cov_int_3_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<int64,3,uint8,1>', ark_cov_int_3_uint8_1, 'StatsMsg', 76);

proc ark_cov_int_3_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<int64,3,uint8,2>', ark_cov_int_3_uint8_2, 'StatsMsg', 76);

proc ark_cov_int_3_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<int64,3,uint8,3>', ark_cov_int_3_uint8_3, 'StatsMsg', 76);

proc ark_cov_int_3_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<int64,3,uint8,4>', ark_cov_int_3_uint8_4, 'StatsMsg', 76);

proc ark_cov_int_3_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<int64,3,float64,1>', ark_cov_int_3_real_1, 'StatsMsg', 76);

proc ark_cov_int_3_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<int64,3,float64,2>', ark_cov_int_3_real_2, 'StatsMsg', 76);

proc ark_cov_int_3_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<int64,3,float64,3>', ark_cov_int_3_real_3, 'StatsMsg', 76);

proc ark_cov_int_3_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<int64,3,float64,4>', ark_cov_int_3_real_4, 'StatsMsg', 76);

proc ark_cov_int_3_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<int64,3,bool,1>', ark_cov_int_3_bool_1, 'StatsMsg', 76);

proc ark_cov_int_3_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<int64,3,bool,2>', ark_cov_int_3_bool_2, 'StatsMsg', 76);

proc ark_cov_int_3_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<int64,3,bool,3>', ark_cov_int_3_bool_3, 'StatsMsg', 76);

proc ark_cov_int_3_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<int64,3,bool,4>', ark_cov_int_3_bool_4, 'StatsMsg', 76);

proc ark_cov_int_3_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<int64,3,bigint,1>', ark_cov_int_3_bigint_1, 'StatsMsg', 76);

proc ark_cov_int_3_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<int64,3,bigint,2>', ark_cov_int_3_bigint_2, 'StatsMsg', 76);

proc ark_cov_int_3_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<int64,3,bigint,3>', ark_cov_int_3_bigint_3, 'StatsMsg', 76);

proc ark_cov_int_3_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<int64,3,bigint,4>', ark_cov_int_3_bigint_4, 'StatsMsg', 76);

proc ark_cov_int_4_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<int64,4,int64,1>', ark_cov_int_4_int_1, 'StatsMsg', 76);

proc ark_cov_int_4_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<int64,4,int64,2>', ark_cov_int_4_int_2, 'StatsMsg', 76);

proc ark_cov_int_4_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<int64,4,int64,3>', ark_cov_int_4_int_3, 'StatsMsg', 76);

proc ark_cov_int_4_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<int64,4,int64,4>', ark_cov_int_4_int_4, 'StatsMsg', 76);

proc ark_cov_int_4_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<int64,4,uint64,1>', ark_cov_int_4_uint_1, 'StatsMsg', 76);

proc ark_cov_int_4_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<int64,4,uint64,2>', ark_cov_int_4_uint_2, 'StatsMsg', 76);

proc ark_cov_int_4_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<int64,4,uint64,3>', ark_cov_int_4_uint_3, 'StatsMsg', 76);

proc ark_cov_int_4_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<int64,4,uint64,4>', ark_cov_int_4_uint_4, 'StatsMsg', 76);

proc ark_cov_int_4_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<int64,4,uint8,1>', ark_cov_int_4_uint8_1, 'StatsMsg', 76);

proc ark_cov_int_4_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<int64,4,uint8,2>', ark_cov_int_4_uint8_2, 'StatsMsg', 76);

proc ark_cov_int_4_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<int64,4,uint8,3>', ark_cov_int_4_uint8_3, 'StatsMsg', 76);

proc ark_cov_int_4_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<int64,4,uint8,4>', ark_cov_int_4_uint8_4, 'StatsMsg', 76);

proc ark_cov_int_4_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<int64,4,float64,1>', ark_cov_int_4_real_1, 'StatsMsg', 76);

proc ark_cov_int_4_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<int64,4,float64,2>', ark_cov_int_4_real_2, 'StatsMsg', 76);

proc ark_cov_int_4_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<int64,4,float64,3>', ark_cov_int_4_real_3, 'StatsMsg', 76);

proc ark_cov_int_4_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<int64,4,float64,4>', ark_cov_int_4_real_4, 'StatsMsg', 76);

proc ark_cov_int_4_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<int64,4,bool,1>', ark_cov_int_4_bool_1, 'StatsMsg', 76);

proc ark_cov_int_4_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<int64,4,bool,2>', ark_cov_int_4_bool_2, 'StatsMsg', 76);

proc ark_cov_int_4_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<int64,4,bool,3>', ark_cov_int_4_bool_3, 'StatsMsg', 76);

proc ark_cov_int_4_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<int64,4,bool,4>', ark_cov_int_4_bool_4, 'StatsMsg', 76);

proc ark_cov_int_4_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<int64,4,bigint,1>', ark_cov_int_4_bigint_1, 'StatsMsg', 76);

proc ark_cov_int_4_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<int64,4,bigint,2>', ark_cov_int_4_bigint_2, 'StatsMsg', 76);

proc ark_cov_int_4_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<int64,4,bigint,3>', ark_cov_int_4_bigint_3, 'StatsMsg', 76);

proc ark_cov_int_4_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<int64,4,bigint,4>', ark_cov_int_4_bigint_4, 'StatsMsg', 76);

proc ark_cov_uint_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<uint64,1,int64,1>', ark_cov_uint_1_int_1, 'StatsMsg', 76);

proc ark_cov_uint_1_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<uint64,1,int64,2>', ark_cov_uint_1_int_2, 'StatsMsg', 76);

proc ark_cov_uint_1_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<uint64,1,int64,3>', ark_cov_uint_1_int_3, 'StatsMsg', 76);

proc ark_cov_uint_1_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<uint64,1,int64,4>', ark_cov_uint_1_int_4, 'StatsMsg', 76);

proc ark_cov_uint_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<uint64,1,uint64,1>', ark_cov_uint_1_uint_1, 'StatsMsg', 76);

proc ark_cov_uint_1_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<uint64,1,uint64,2>', ark_cov_uint_1_uint_2, 'StatsMsg', 76);

proc ark_cov_uint_1_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<uint64,1,uint64,3>', ark_cov_uint_1_uint_3, 'StatsMsg', 76);

proc ark_cov_uint_1_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<uint64,1,uint64,4>', ark_cov_uint_1_uint_4, 'StatsMsg', 76);

proc ark_cov_uint_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<uint64,1,uint8,1>', ark_cov_uint_1_uint8_1, 'StatsMsg', 76);

proc ark_cov_uint_1_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<uint64,1,uint8,2>', ark_cov_uint_1_uint8_2, 'StatsMsg', 76);

proc ark_cov_uint_1_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<uint64,1,uint8,3>', ark_cov_uint_1_uint8_3, 'StatsMsg', 76);

proc ark_cov_uint_1_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<uint64,1,uint8,4>', ark_cov_uint_1_uint8_4, 'StatsMsg', 76);

proc ark_cov_uint_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<uint64,1,float64,1>', ark_cov_uint_1_real_1, 'StatsMsg', 76);

proc ark_cov_uint_1_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<uint64,1,float64,2>', ark_cov_uint_1_real_2, 'StatsMsg', 76);

proc ark_cov_uint_1_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<uint64,1,float64,3>', ark_cov_uint_1_real_3, 'StatsMsg', 76);

proc ark_cov_uint_1_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<uint64,1,float64,4>', ark_cov_uint_1_real_4, 'StatsMsg', 76);

proc ark_cov_uint_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<uint64,1,bool,1>', ark_cov_uint_1_bool_1, 'StatsMsg', 76);

proc ark_cov_uint_1_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<uint64,1,bool,2>', ark_cov_uint_1_bool_2, 'StatsMsg', 76);

proc ark_cov_uint_1_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<uint64,1,bool,3>', ark_cov_uint_1_bool_3, 'StatsMsg', 76);

proc ark_cov_uint_1_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<uint64,1,bool,4>', ark_cov_uint_1_bool_4, 'StatsMsg', 76);

proc ark_cov_uint_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<uint64,1,bigint,1>', ark_cov_uint_1_bigint_1, 'StatsMsg', 76);

proc ark_cov_uint_1_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<uint64,1,bigint,2>', ark_cov_uint_1_bigint_2, 'StatsMsg', 76);

proc ark_cov_uint_1_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<uint64,1,bigint,3>', ark_cov_uint_1_bigint_3, 'StatsMsg', 76);

proc ark_cov_uint_1_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<uint64,1,bigint,4>', ark_cov_uint_1_bigint_4, 'StatsMsg', 76);

proc ark_cov_uint_2_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<uint64,2,int64,1>', ark_cov_uint_2_int_1, 'StatsMsg', 76);

proc ark_cov_uint_2_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<uint64,2,int64,2>', ark_cov_uint_2_int_2, 'StatsMsg', 76);

proc ark_cov_uint_2_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<uint64,2,int64,3>', ark_cov_uint_2_int_3, 'StatsMsg', 76);

proc ark_cov_uint_2_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<uint64,2,int64,4>', ark_cov_uint_2_int_4, 'StatsMsg', 76);

proc ark_cov_uint_2_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<uint64,2,uint64,1>', ark_cov_uint_2_uint_1, 'StatsMsg', 76);

proc ark_cov_uint_2_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<uint64,2,uint64,2>', ark_cov_uint_2_uint_2, 'StatsMsg', 76);

proc ark_cov_uint_2_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<uint64,2,uint64,3>', ark_cov_uint_2_uint_3, 'StatsMsg', 76);

proc ark_cov_uint_2_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<uint64,2,uint64,4>', ark_cov_uint_2_uint_4, 'StatsMsg', 76);

proc ark_cov_uint_2_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<uint64,2,uint8,1>', ark_cov_uint_2_uint8_1, 'StatsMsg', 76);

proc ark_cov_uint_2_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<uint64,2,uint8,2>', ark_cov_uint_2_uint8_2, 'StatsMsg', 76);

proc ark_cov_uint_2_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<uint64,2,uint8,3>', ark_cov_uint_2_uint8_3, 'StatsMsg', 76);

proc ark_cov_uint_2_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<uint64,2,uint8,4>', ark_cov_uint_2_uint8_4, 'StatsMsg', 76);

proc ark_cov_uint_2_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<uint64,2,float64,1>', ark_cov_uint_2_real_1, 'StatsMsg', 76);

proc ark_cov_uint_2_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<uint64,2,float64,2>', ark_cov_uint_2_real_2, 'StatsMsg', 76);

proc ark_cov_uint_2_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<uint64,2,float64,3>', ark_cov_uint_2_real_3, 'StatsMsg', 76);

proc ark_cov_uint_2_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<uint64,2,float64,4>', ark_cov_uint_2_real_4, 'StatsMsg', 76);

proc ark_cov_uint_2_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<uint64,2,bool,1>', ark_cov_uint_2_bool_1, 'StatsMsg', 76);

proc ark_cov_uint_2_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<uint64,2,bool,2>', ark_cov_uint_2_bool_2, 'StatsMsg', 76);

proc ark_cov_uint_2_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<uint64,2,bool,3>', ark_cov_uint_2_bool_3, 'StatsMsg', 76);

proc ark_cov_uint_2_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<uint64,2,bool,4>', ark_cov_uint_2_bool_4, 'StatsMsg', 76);

proc ark_cov_uint_2_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<uint64,2,bigint,1>', ark_cov_uint_2_bigint_1, 'StatsMsg', 76);

proc ark_cov_uint_2_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<uint64,2,bigint,2>', ark_cov_uint_2_bigint_2, 'StatsMsg', 76);

proc ark_cov_uint_2_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<uint64,2,bigint,3>', ark_cov_uint_2_bigint_3, 'StatsMsg', 76);

proc ark_cov_uint_2_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<uint64,2,bigint,4>', ark_cov_uint_2_bigint_4, 'StatsMsg', 76);

proc ark_cov_uint_3_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<uint64,3,int64,1>', ark_cov_uint_3_int_1, 'StatsMsg', 76);

proc ark_cov_uint_3_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<uint64,3,int64,2>', ark_cov_uint_3_int_2, 'StatsMsg', 76);

proc ark_cov_uint_3_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<uint64,3,int64,3>', ark_cov_uint_3_int_3, 'StatsMsg', 76);

proc ark_cov_uint_3_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<uint64,3,int64,4>', ark_cov_uint_3_int_4, 'StatsMsg', 76);

proc ark_cov_uint_3_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<uint64,3,uint64,1>', ark_cov_uint_3_uint_1, 'StatsMsg', 76);

proc ark_cov_uint_3_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<uint64,3,uint64,2>', ark_cov_uint_3_uint_2, 'StatsMsg', 76);

proc ark_cov_uint_3_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<uint64,3,uint64,3>', ark_cov_uint_3_uint_3, 'StatsMsg', 76);

proc ark_cov_uint_3_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<uint64,3,uint64,4>', ark_cov_uint_3_uint_4, 'StatsMsg', 76);

proc ark_cov_uint_3_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<uint64,3,uint8,1>', ark_cov_uint_3_uint8_1, 'StatsMsg', 76);

proc ark_cov_uint_3_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<uint64,3,uint8,2>', ark_cov_uint_3_uint8_2, 'StatsMsg', 76);

proc ark_cov_uint_3_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<uint64,3,uint8,3>', ark_cov_uint_3_uint8_3, 'StatsMsg', 76);

proc ark_cov_uint_3_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<uint64,3,uint8,4>', ark_cov_uint_3_uint8_4, 'StatsMsg', 76);

proc ark_cov_uint_3_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<uint64,3,float64,1>', ark_cov_uint_3_real_1, 'StatsMsg', 76);

proc ark_cov_uint_3_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<uint64,3,float64,2>', ark_cov_uint_3_real_2, 'StatsMsg', 76);

proc ark_cov_uint_3_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<uint64,3,float64,3>', ark_cov_uint_3_real_3, 'StatsMsg', 76);

proc ark_cov_uint_3_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<uint64,3,float64,4>', ark_cov_uint_3_real_4, 'StatsMsg', 76);

proc ark_cov_uint_3_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<uint64,3,bool,1>', ark_cov_uint_3_bool_1, 'StatsMsg', 76);

proc ark_cov_uint_3_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<uint64,3,bool,2>', ark_cov_uint_3_bool_2, 'StatsMsg', 76);

proc ark_cov_uint_3_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<uint64,3,bool,3>', ark_cov_uint_3_bool_3, 'StatsMsg', 76);

proc ark_cov_uint_3_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<uint64,3,bool,4>', ark_cov_uint_3_bool_4, 'StatsMsg', 76);

proc ark_cov_uint_3_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<uint64,3,bigint,1>', ark_cov_uint_3_bigint_1, 'StatsMsg', 76);

proc ark_cov_uint_3_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<uint64,3,bigint,2>', ark_cov_uint_3_bigint_2, 'StatsMsg', 76);

proc ark_cov_uint_3_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<uint64,3,bigint,3>', ark_cov_uint_3_bigint_3, 'StatsMsg', 76);

proc ark_cov_uint_3_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<uint64,3,bigint,4>', ark_cov_uint_3_bigint_4, 'StatsMsg', 76);

proc ark_cov_uint_4_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<uint64,4,int64,1>', ark_cov_uint_4_int_1, 'StatsMsg', 76);

proc ark_cov_uint_4_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<uint64,4,int64,2>', ark_cov_uint_4_int_2, 'StatsMsg', 76);

proc ark_cov_uint_4_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<uint64,4,int64,3>', ark_cov_uint_4_int_3, 'StatsMsg', 76);

proc ark_cov_uint_4_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<uint64,4,int64,4>', ark_cov_uint_4_int_4, 'StatsMsg', 76);

proc ark_cov_uint_4_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<uint64,4,uint64,1>', ark_cov_uint_4_uint_1, 'StatsMsg', 76);

proc ark_cov_uint_4_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<uint64,4,uint64,2>', ark_cov_uint_4_uint_2, 'StatsMsg', 76);

proc ark_cov_uint_4_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<uint64,4,uint64,3>', ark_cov_uint_4_uint_3, 'StatsMsg', 76);

proc ark_cov_uint_4_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<uint64,4,uint64,4>', ark_cov_uint_4_uint_4, 'StatsMsg', 76);

proc ark_cov_uint_4_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<uint64,4,uint8,1>', ark_cov_uint_4_uint8_1, 'StatsMsg', 76);

proc ark_cov_uint_4_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<uint64,4,uint8,2>', ark_cov_uint_4_uint8_2, 'StatsMsg', 76);

proc ark_cov_uint_4_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<uint64,4,uint8,3>', ark_cov_uint_4_uint8_3, 'StatsMsg', 76);

proc ark_cov_uint_4_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<uint64,4,uint8,4>', ark_cov_uint_4_uint8_4, 'StatsMsg', 76);

proc ark_cov_uint_4_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<uint64,4,float64,1>', ark_cov_uint_4_real_1, 'StatsMsg', 76);

proc ark_cov_uint_4_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<uint64,4,float64,2>', ark_cov_uint_4_real_2, 'StatsMsg', 76);

proc ark_cov_uint_4_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<uint64,4,float64,3>', ark_cov_uint_4_real_3, 'StatsMsg', 76);

proc ark_cov_uint_4_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<uint64,4,float64,4>', ark_cov_uint_4_real_4, 'StatsMsg', 76);

proc ark_cov_uint_4_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<uint64,4,bool,1>', ark_cov_uint_4_bool_1, 'StatsMsg', 76);

proc ark_cov_uint_4_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<uint64,4,bool,2>', ark_cov_uint_4_bool_2, 'StatsMsg', 76);

proc ark_cov_uint_4_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<uint64,4,bool,3>', ark_cov_uint_4_bool_3, 'StatsMsg', 76);

proc ark_cov_uint_4_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<uint64,4,bool,4>', ark_cov_uint_4_bool_4, 'StatsMsg', 76);

proc ark_cov_uint_4_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<uint64,4,bigint,1>', ark_cov_uint_4_bigint_1, 'StatsMsg', 76);

proc ark_cov_uint_4_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<uint64,4,bigint,2>', ark_cov_uint_4_bigint_2, 'StatsMsg', 76);

proc ark_cov_uint_4_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<uint64,4,bigint,3>', ark_cov_uint_4_bigint_3, 'StatsMsg', 76);

proc ark_cov_uint_4_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<uint64,4,bigint,4>', ark_cov_uint_4_bigint_4, 'StatsMsg', 76);

proc ark_cov_uint8_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<uint8,1,int64,1>', ark_cov_uint8_1_int_1, 'StatsMsg', 76);

proc ark_cov_uint8_1_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<uint8,1,int64,2>', ark_cov_uint8_1_int_2, 'StatsMsg', 76);

proc ark_cov_uint8_1_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<uint8,1,int64,3>', ark_cov_uint8_1_int_3, 'StatsMsg', 76);

proc ark_cov_uint8_1_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<uint8,1,int64,4>', ark_cov_uint8_1_int_4, 'StatsMsg', 76);

proc ark_cov_uint8_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<uint8,1,uint64,1>', ark_cov_uint8_1_uint_1, 'StatsMsg', 76);

proc ark_cov_uint8_1_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<uint8,1,uint64,2>', ark_cov_uint8_1_uint_2, 'StatsMsg', 76);

proc ark_cov_uint8_1_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<uint8,1,uint64,3>', ark_cov_uint8_1_uint_3, 'StatsMsg', 76);

proc ark_cov_uint8_1_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<uint8,1,uint64,4>', ark_cov_uint8_1_uint_4, 'StatsMsg', 76);

proc ark_cov_uint8_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<uint8,1,uint8,1>', ark_cov_uint8_1_uint8_1, 'StatsMsg', 76);

proc ark_cov_uint8_1_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<uint8,1,uint8,2>', ark_cov_uint8_1_uint8_2, 'StatsMsg', 76);

proc ark_cov_uint8_1_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<uint8,1,uint8,3>', ark_cov_uint8_1_uint8_3, 'StatsMsg', 76);

proc ark_cov_uint8_1_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<uint8,1,uint8,4>', ark_cov_uint8_1_uint8_4, 'StatsMsg', 76);

proc ark_cov_uint8_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<uint8,1,float64,1>', ark_cov_uint8_1_real_1, 'StatsMsg', 76);

proc ark_cov_uint8_1_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<uint8,1,float64,2>', ark_cov_uint8_1_real_2, 'StatsMsg', 76);

proc ark_cov_uint8_1_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<uint8,1,float64,3>', ark_cov_uint8_1_real_3, 'StatsMsg', 76);

proc ark_cov_uint8_1_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<uint8,1,float64,4>', ark_cov_uint8_1_real_4, 'StatsMsg', 76);

proc ark_cov_uint8_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<uint8,1,bool,1>', ark_cov_uint8_1_bool_1, 'StatsMsg', 76);

proc ark_cov_uint8_1_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<uint8,1,bool,2>', ark_cov_uint8_1_bool_2, 'StatsMsg', 76);

proc ark_cov_uint8_1_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<uint8,1,bool,3>', ark_cov_uint8_1_bool_3, 'StatsMsg', 76);

proc ark_cov_uint8_1_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<uint8,1,bool,4>', ark_cov_uint8_1_bool_4, 'StatsMsg', 76);

proc ark_cov_uint8_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<uint8,1,bigint,1>', ark_cov_uint8_1_bigint_1, 'StatsMsg', 76);

proc ark_cov_uint8_1_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<uint8,1,bigint,2>', ark_cov_uint8_1_bigint_2, 'StatsMsg', 76);

proc ark_cov_uint8_1_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<uint8,1,bigint,3>', ark_cov_uint8_1_bigint_3, 'StatsMsg', 76);

proc ark_cov_uint8_1_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<uint8,1,bigint,4>', ark_cov_uint8_1_bigint_4, 'StatsMsg', 76);

proc ark_cov_uint8_2_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<uint8,2,int64,1>', ark_cov_uint8_2_int_1, 'StatsMsg', 76);

proc ark_cov_uint8_2_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<uint8,2,int64,2>', ark_cov_uint8_2_int_2, 'StatsMsg', 76);

proc ark_cov_uint8_2_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<uint8,2,int64,3>', ark_cov_uint8_2_int_3, 'StatsMsg', 76);

proc ark_cov_uint8_2_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<uint8,2,int64,4>', ark_cov_uint8_2_int_4, 'StatsMsg', 76);

proc ark_cov_uint8_2_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<uint8,2,uint64,1>', ark_cov_uint8_2_uint_1, 'StatsMsg', 76);

proc ark_cov_uint8_2_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<uint8,2,uint64,2>', ark_cov_uint8_2_uint_2, 'StatsMsg', 76);

proc ark_cov_uint8_2_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<uint8,2,uint64,3>', ark_cov_uint8_2_uint_3, 'StatsMsg', 76);

proc ark_cov_uint8_2_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<uint8,2,uint64,4>', ark_cov_uint8_2_uint_4, 'StatsMsg', 76);

proc ark_cov_uint8_2_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<uint8,2,uint8,1>', ark_cov_uint8_2_uint8_1, 'StatsMsg', 76);

proc ark_cov_uint8_2_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<uint8,2,uint8,2>', ark_cov_uint8_2_uint8_2, 'StatsMsg', 76);

proc ark_cov_uint8_2_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<uint8,2,uint8,3>', ark_cov_uint8_2_uint8_3, 'StatsMsg', 76);

proc ark_cov_uint8_2_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<uint8,2,uint8,4>', ark_cov_uint8_2_uint8_4, 'StatsMsg', 76);

proc ark_cov_uint8_2_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<uint8,2,float64,1>', ark_cov_uint8_2_real_1, 'StatsMsg', 76);

proc ark_cov_uint8_2_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<uint8,2,float64,2>', ark_cov_uint8_2_real_2, 'StatsMsg', 76);

proc ark_cov_uint8_2_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<uint8,2,float64,3>', ark_cov_uint8_2_real_3, 'StatsMsg', 76);

proc ark_cov_uint8_2_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<uint8,2,float64,4>', ark_cov_uint8_2_real_4, 'StatsMsg', 76);

proc ark_cov_uint8_2_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<uint8,2,bool,1>', ark_cov_uint8_2_bool_1, 'StatsMsg', 76);

proc ark_cov_uint8_2_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<uint8,2,bool,2>', ark_cov_uint8_2_bool_2, 'StatsMsg', 76);

proc ark_cov_uint8_2_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<uint8,2,bool,3>', ark_cov_uint8_2_bool_3, 'StatsMsg', 76);

proc ark_cov_uint8_2_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<uint8,2,bool,4>', ark_cov_uint8_2_bool_4, 'StatsMsg', 76);

proc ark_cov_uint8_2_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<uint8,2,bigint,1>', ark_cov_uint8_2_bigint_1, 'StatsMsg', 76);

proc ark_cov_uint8_2_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<uint8,2,bigint,2>', ark_cov_uint8_2_bigint_2, 'StatsMsg', 76);

proc ark_cov_uint8_2_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<uint8,2,bigint,3>', ark_cov_uint8_2_bigint_3, 'StatsMsg', 76);

proc ark_cov_uint8_2_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<uint8,2,bigint,4>', ark_cov_uint8_2_bigint_4, 'StatsMsg', 76);

proc ark_cov_uint8_3_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<uint8,3,int64,1>', ark_cov_uint8_3_int_1, 'StatsMsg', 76);

proc ark_cov_uint8_3_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<uint8,3,int64,2>', ark_cov_uint8_3_int_2, 'StatsMsg', 76);

proc ark_cov_uint8_3_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<uint8,3,int64,3>', ark_cov_uint8_3_int_3, 'StatsMsg', 76);

proc ark_cov_uint8_3_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<uint8,3,int64,4>', ark_cov_uint8_3_int_4, 'StatsMsg', 76);

proc ark_cov_uint8_3_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<uint8,3,uint64,1>', ark_cov_uint8_3_uint_1, 'StatsMsg', 76);

proc ark_cov_uint8_3_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<uint8,3,uint64,2>', ark_cov_uint8_3_uint_2, 'StatsMsg', 76);

proc ark_cov_uint8_3_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<uint8,3,uint64,3>', ark_cov_uint8_3_uint_3, 'StatsMsg', 76);

proc ark_cov_uint8_3_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<uint8,3,uint64,4>', ark_cov_uint8_3_uint_4, 'StatsMsg', 76);

proc ark_cov_uint8_3_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<uint8,3,uint8,1>', ark_cov_uint8_3_uint8_1, 'StatsMsg', 76);

proc ark_cov_uint8_3_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<uint8,3,uint8,2>', ark_cov_uint8_3_uint8_2, 'StatsMsg', 76);

proc ark_cov_uint8_3_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<uint8,3,uint8,3>', ark_cov_uint8_3_uint8_3, 'StatsMsg', 76);

proc ark_cov_uint8_3_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<uint8,3,uint8,4>', ark_cov_uint8_3_uint8_4, 'StatsMsg', 76);

proc ark_cov_uint8_3_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<uint8,3,float64,1>', ark_cov_uint8_3_real_1, 'StatsMsg', 76);

proc ark_cov_uint8_3_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<uint8,3,float64,2>', ark_cov_uint8_3_real_2, 'StatsMsg', 76);

proc ark_cov_uint8_3_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<uint8,3,float64,3>', ark_cov_uint8_3_real_3, 'StatsMsg', 76);

proc ark_cov_uint8_3_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<uint8,3,float64,4>', ark_cov_uint8_3_real_4, 'StatsMsg', 76);

proc ark_cov_uint8_3_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<uint8,3,bool,1>', ark_cov_uint8_3_bool_1, 'StatsMsg', 76);

proc ark_cov_uint8_3_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<uint8,3,bool,2>', ark_cov_uint8_3_bool_2, 'StatsMsg', 76);

proc ark_cov_uint8_3_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<uint8,3,bool,3>', ark_cov_uint8_3_bool_3, 'StatsMsg', 76);

proc ark_cov_uint8_3_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<uint8,3,bool,4>', ark_cov_uint8_3_bool_4, 'StatsMsg', 76);

proc ark_cov_uint8_3_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<uint8,3,bigint,1>', ark_cov_uint8_3_bigint_1, 'StatsMsg', 76);

proc ark_cov_uint8_3_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<uint8,3,bigint,2>', ark_cov_uint8_3_bigint_2, 'StatsMsg', 76);

proc ark_cov_uint8_3_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<uint8,3,bigint,3>', ark_cov_uint8_3_bigint_3, 'StatsMsg', 76);

proc ark_cov_uint8_3_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<uint8,3,bigint,4>', ark_cov_uint8_3_bigint_4, 'StatsMsg', 76);

proc ark_cov_uint8_4_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<uint8,4,int64,1>', ark_cov_uint8_4_int_1, 'StatsMsg', 76);

proc ark_cov_uint8_4_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<uint8,4,int64,2>', ark_cov_uint8_4_int_2, 'StatsMsg', 76);

proc ark_cov_uint8_4_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<uint8,4,int64,3>', ark_cov_uint8_4_int_3, 'StatsMsg', 76);

proc ark_cov_uint8_4_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<uint8,4,int64,4>', ark_cov_uint8_4_int_4, 'StatsMsg', 76);

proc ark_cov_uint8_4_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<uint8,4,uint64,1>', ark_cov_uint8_4_uint_1, 'StatsMsg', 76);

proc ark_cov_uint8_4_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<uint8,4,uint64,2>', ark_cov_uint8_4_uint_2, 'StatsMsg', 76);

proc ark_cov_uint8_4_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<uint8,4,uint64,3>', ark_cov_uint8_4_uint_3, 'StatsMsg', 76);

proc ark_cov_uint8_4_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<uint8,4,uint64,4>', ark_cov_uint8_4_uint_4, 'StatsMsg', 76);

proc ark_cov_uint8_4_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<uint8,4,uint8,1>', ark_cov_uint8_4_uint8_1, 'StatsMsg', 76);

proc ark_cov_uint8_4_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<uint8,4,uint8,2>', ark_cov_uint8_4_uint8_2, 'StatsMsg', 76);

proc ark_cov_uint8_4_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<uint8,4,uint8,3>', ark_cov_uint8_4_uint8_3, 'StatsMsg', 76);

proc ark_cov_uint8_4_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<uint8,4,uint8,4>', ark_cov_uint8_4_uint8_4, 'StatsMsg', 76);

proc ark_cov_uint8_4_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<uint8,4,float64,1>', ark_cov_uint8_4_real_1, 'StatsMsg', 76);

proc ark_cov_uint8_4_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<uint8,4,float64,2>', ark_cov_uint8_4_real_2, 'StatsMsg', 76);

proc ark_cov_uint8_4_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<uint8,4,float64,3>', ark_cov_uint8_4_real_3, 'StatsMsg', 76);

proc ark_cov_uint8_4_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<uint8,4,float64,4>', ark_cov_uint8_4_real_4, 'StatsMsg', 76);

proc ark_cov_uint8_4_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<uint8,4,bool,1>', ark_cov_uint8_4_bool_1, 'StatsMsg', 76);

proc ark_cov_uint8_4_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<uint8,4,bool,2>', ark_cov_uint8_4_bool_2, 'StatsMsg', 76);

proc ark_cov_uint8_4_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<uint8,4,bool,3>', ark_cov_uint8_4_bool_3, 'StatsMsg', 76);

proc ark_cov_uint8_4_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<uint8,4,bool,4>', ark_cov_uint8_4_bool_4, 'StatsMsg', 76);

proc ark_cov_uint8_4_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<uint8,4,bigint,1>', ark_cov_uint8_4_bigint_1, 'StatsMsg', 76);

proc ark_cov_uint8_4_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<uint8,4,bigint,2>', ark_cov_uint8_4_bigint_2, 'StatsMsg', 76);

proc ark_cov_uint8_4_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<uint8,4,bigint,3>', ark_cov_uint8_4_bigint_3, 'StatsMsg', 76);

proc ark_cov_uint8_4_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<uint8,4,bigint,4>', ark_cov_uint8_4_bigint_4, 'StatsMsg', 76);

proc ark_cov_real_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<float64,1,int64,1>', ark_cov_real_1_int_1, 'StatsMsg', 76);

proc ark_cov_real_1_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<float64,1,int64,2>', ark_cov_real_1_int_2, 'StatsMsg', 76);

proc ark_cov_real_1_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<float64,1,int64,3>', ark_cov_real_1_int_3, 'StatsMsg', 76);

proc ark_cov_real_1_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<float64,1,int64,4>', ark_cov_real_1_int_4, 'StatsMsg', 76);

proc ark_cov_real_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<float64,1,uint64,1>', ark_cov_real_1_uint_1, 'StatsMsg', 76);

proc ark_cov_real_1_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<float64,1,uint64,2>', ark_cov_real_1_uint_2, 'StatsMsg', 76);

proc ark_cov_real_1_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<float64,1,uint64,3>', ark_cov_real_1_uint_3, 'StatsMsg', 76);

proc ark_cov_real_1_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<float64,1,uint64,4>', ark_cov_real_1_uint_4, 'StatsMsg', 76);

proc ark_cov_real_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<float64,1,uint8,1>', ark_cov_real_1_uint8_1, 'StatsMsg', 76);

proc ark_cov_real_1_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<float64,1,uint8,2>', ark_cov_real_1_uint8_2, 'StatsMsg', 76);

proc ark_cov_real_1_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<float64,1,uint8,3>', ark_cov_real_1_uint8_3, 'StatsMsg', 76);

proc ark_cov_real_1_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<float64,1,uint8,4>', ark_cov_real_1_uint8_4, 'StatsMsg', 76);

proc ark_cov_real_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<float64,1,float64,1>', ark_cov_real_1_real_1, 'StatsMsg', 76);

proc ark_cov_real_1_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<float64,1,float64,2>', ark_cov_real_1_real_2, 'StatsMsg', 76);

proc ark_cov_real_1_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<float64,1,float64,3>', ark_cov_real_1_real_3, 'StatsMsg', 76);

proc ark_cov_real_1_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<float64,1,float64,4>', ark_cov_real_1_real_4, 'StatsMsg', 76);

proc ark_cov_real_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<float64,1,bool,1>', ark_cov_real_1_bool_1, 'StatsMsg', 76);

proc ark_cov_real_1_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<float64,1,bool,2>', ark_cov_real_1_bool_2, 'StatsMsg', 76);

proc ark_cov_real_1_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<float64,1,bool,3>', ark_cov_real_1_bool_3, 'StatsMsg', 76);

proc ark_cov_real_1_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<float64,1,bool,4>', ark_cov_real_1_bool_4, 'StatsMsg', 76);

proc ark_cov_real_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<float64,1,bigint,1>', ark_cov_real_1_bigint_1, 'StatsMsg', 76);

proc ark_cov_real_1_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<float64,1,bigint,2>', ark_cov_real_1_bigint_2, 'StatsMsg', 76);

proc ark_cov_real_1_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<float64,1,bigint,3>', ark_cov_real_1_bigint_3, 'StatsMsg', 76);

proc ark_cov_real_1_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<float64,1,bigint,4>', ark_cov_real_1_bigint_4, 'StatsMsg', 76);

proc ark_cov_real_2_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<float64,2,int64,1>', ark_cov_real_2_int_1, 'StatsMsg', 76);

proc ark_cov_real_2_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<float64,2,int64,2>', ark_cov_real_2_int_2, 'StatsMsg', 76);

proc ark_cov_real_2_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<float64,2,int64,3>', ark_cov_real_2_int_3, 'StatsMsg', 76);

proc ark_cov_real_2_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<float64,2,int64,4>', ark_cov_real_2_int_4, 'StatsMsg', 76);

proc ark_cov_real_2_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<float64,2,uint64,1>', ark_cov_real_2_uint_1, 'StatsMsg', 76);

proc ark_cov_real_2_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<float64,2,uint64,2>', ark_cov_real_2_uint_2, 'StatsMsg', 76);

proc ark_cov_real_2_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<float64,2,uint64,3>', ark_cov_real_2_uint_3, 'StatsMsg', 76);

proc ark_cov_real_2_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<float64,2,uint64,4>', ark_cov_real_2_uint_4, 'StatsMsg', 76);

proc ark_cov_real_2_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<float64,2,uint8,1>', ark_cov_real_2_uint8_1, 'StatsMsg', 76);

proc ark_cov_real_2_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<float64,2,uint8,2>', ark_cov_real_2_uint8_2, 'StatsMsg', 76);

proc ark_cov_real_2_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<float64,2,uint8,3>', ark_cov_real_2_uint8_3, 'StatsMsg', 76);

proc ark_cov_real_2_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<float64,2,uint8,4>', ark_cov_real_2_uint8_4, 'StatsMsg', 76);

proc ark_cov_real_2_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<float64,2,float64,1>', ark_cov_real_2_real_1, 'StatsMsg', 76);

proc ark_cov_real_2_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<float64,2,float64,2>', ark_cov_real_2_real_2, 'StatsMsg', 76);

proc ark_cov_real_2_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<float64,2,float64,3>', ark_cov_real_2_real_3, 'StatsMsg', 76);

proc ark_cov_real_2_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<float64,2,float64,4>', ark_cov_real_2_real_4, 'StatsMsg', 76);

proc ark_cov_real_2_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<float64,2,bool,1>', ark_cov_real_2_bool_1, 'StatsMsg', 76);

proc ark_cov_real_2_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<float64,2,bool,2>', ark_cov_real_2_bool_2, 'StatsMsg', 76);

proc ark_cov_real_2_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<float64,2,bool,3>', ark_cov_real_2_bool_3, 'StatsMsg', 76);

proc ark_cov_real_2_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<float64,2,bool,4>', ark_cov_real_2_bool_4, 'StatsMsg', 76);

proc ark_cov_real_2_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<float64,2,bigint,1>', ark_cov_real_2_bigint_1, 'StatsMsg', 76);

proc ark_cov_real_2_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<float64,2,bigint,2>', ark_cov_real_2_bigint_2, 'StatsMsg', 76);

proc ark_cov_real_2_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<float64,2,bigint,3>', ark_cov_real_2_bigint_3, 'StatsMsg', 76);

proc ark_cov_real_2_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<float64,2,bigint,4>', ark_cov_real_2_bigint_4, 'StatsMsg', 76);

proc ark_cov_real_3_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<float64,3,int64,1>', ark_cov_real_3_int_1, 'StatsMsg', 76);

proc ark_cov_real_3_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<float64,3,int64,2>', ark_cov_real_3_int_2, 'StatsMsg', 76);

proc ark_cov_real_3_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<float64,3,int64,3>', ark_cov_real_3_int_3, 'StatsMsg', 76);

proc ark_cov_real_3_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<float64,3,int64,4>', ark_cov_real_3_int_4, 'StatsMsg', 76);

proc ark_cov_real_3_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<float64,3,uint64,1>', ark_cov_real_3_uint_1, 'StatsMsg', 76);

proc ark_cov_real_3_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<float64,3,uint64,2>', ark_cov_real_3_uint_2, 'StatsMsg', 76);

proc ark_cov_real_3_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<float64,3,uint64,3>', ark_cov_real_3_uint_3, 'StatsMsg', 76);

proc ark_cov_real_3_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<float64,3,uint64,4>', ark_cov_real_3_uint_4, 'StatsMsg', 76);

proc ark_cov_real_3_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<float64,3,uint8,1>', ark_cov_real_3_uint8_1, 'StatsMsg', 76);

proc ark_cov_real_3_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<float64,3,uint8,2>', ark_cov_real_3_uint8_2, 'StatsMsg', 76);

proc ark_cov_real_3_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<float64,3,uint8,3>', ark_cov_real_3_uint8_3, 'StatsMsg', 76);

proc ark_cov_real_3_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<float64,3,uint8,4>', ark_cov_real_3_uint8_4, 'StatsMsg', 76);

proc ark_cov_real_3_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<float64,3,float64,1>', ark_cov_real_3_real_1, 'StatsMsg', 76);

proc ark_cov_real_3_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<float64,3,float64,2>', ark_cov_real_3_real_2, 'StatsMsg', 76);

proc ark_cov_real_3_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<float64,3,float64,3>', ark_cov_real_3_real_3, 'StatsMsg', 76);

proc ark_cov_real_3_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<float64,3,float64,4>', ark_cov_real_3_real_4, 'StatsMsg', 76);

proc ark_cov_real_3_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<float64,3,bool,1>', ark_cov_real_3_bool_1, 'StatsMsg', 76);

proc ark_cov_real_3_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<float64,3,bool,2>', ark_cov_real_3_bool_2, 'StatsMsg', 76);

proc ark_cov_real_3_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<float64,3,bool,3>', ark_cov_real_3_bool_3, 'StatsMsg', 76);

proc ark_cov_real_3_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<float64,3,bool,4>', ark_cov_real_3_bool_4, 'StatsMsg', 76);

proc ark_cov_real_3_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<float64,3,bigint,1>', ark_cov_real_3_bigint_1, 'StatsMsg', 76);

proc ark_cov_real_3_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<float64,3,bigint,2>', ark_cov_real_3_bigint_2, 'StatsMsg', 76);

proc ark_cov_real_3_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<float64,3,bigint,3>', ark_cov_real_3_bigint_3, 'StatsMsg', 76);

proc ark_cov_real_3_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<float64,3,bigint,4>', ark_cov_real_3_bigint_4, 'StatsMsg', 76);

proc ark_cov_real_4_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<float64,4,int64,1>', ark_cov_real_4_int_1, 'StatsMsg', 76);

proc ark_cov_real_4_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<float64,4,int64,2>', ark_cov_real_4_int_2, 'StatsMsg', 76);

proc ark_cov_real_4_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<float64,4,int64,3>', ark_cov_real_4_int_3, 'StatsMsg', 76);

proc ark_cov_real_4_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<float64,4,int64,4>', ark_cov_real_4_int_4, 'StatsMsg', 76);

proc ark_cov_real_4_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<float64,4,uint64,1>', ark_cov_real_4_uint_1, 'StatsMsg', 76);

proc ark_cov_real_4_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<float64,4,uint64,2>', ark_cov_real_4_uint_2, 'StatsMsg', 76);

proc ark_cov_real_4_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<float64,4,uint64,3>', ark_cov_real_4_uint_3, 'StatsMsg', 76);

proc ark_cov_real_4_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<float64,4,uint64,4>', ark_cov_real_4_uint_4, 'StatsMsg', 76);

proc ark_cov_real_4_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<float64,4,uint8,1>', ark_cov_real_4_uint8_1, 'StatsMsg', 76);

proc ark_cov_real_4_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<float64,4,uint8,2>', ark_cov_real_4_uint8_2, 'StatsMsg', 76);

proc ark_cov_real_4_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<float64,4,uint8,3>', ark_cov_real_4_uint8_3, 'StatsMsg', 76);

proc ark_cov_real_4_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<float64,4,uint8,4>', ark_cov_real_4_uint8_4, 'StatsMsg', 76);

proc ark_cov_real_4_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<float64,4,float64,1>', ark_cov_real_4_real_1, 'StatsMsg', 76);

proc ark_cov_real_4_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<float64,4,float64,2>', ark_cov_real_4_real_2, 'StatsMsg', 76);

proc ark_cov_real_4_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<float64,4,float64,3>', ark_cov_real_4_real_3, 'StatsMsg', 76);

proc ark_cov_real_4_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<float64,4,float64,4>', ark_cov_real_4_real_4, 'StatsMsg', 76);

proc ark_cov_real_4_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<float64,4,bool,1>', ark_cov_real_4_bool_1, 'StatsMsg', 76);

proc ark_cov_real_4_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<float64,4,bool,2>', ark_cov_real_4_bool_2, 'StatsMsg', 76);

proc ark_cov_real_4_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<float64,4,bool,3>', ark_cov_real_4_bool_3, 'StatsMsg', 76);

proc ark_cov_real_4_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<float64,4,bool,4>', ark_cov_real_4_bool_4, 'StatsMsg', 76);

proc ark_cov_real_4_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<float64,4,bigint,1>', ark_cov_real_4_bigint_1, 'StatsMsg', 76);

proc ark_cov_real_4_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<float64,4,bigint,2>', ark_cov_real_4_bigint_2, 'StatsMsg', 76);

proc ark_cov_real_4_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<float64,4,bigint,3>', ark_cov_real_4_bigint_3, 'StatsMsg', 76);

proc ark_cov_real_4_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<float64,4,bigint,4>', ark_cov_real_4_bigint_4, 'StatsMsg', 76);

proc ark_cov_bool_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<bool,1,int64,1>', ark_cov_bool_1_int_1, 'StatsMsg', 76);

proc ark_cov_bool_1_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<bool,1,int64,2>', ark_cov_bool_1_int_2, 'StatsMsg', 76);

proc ark_cov_bool_1_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<bool,1,int64,3>', ark_cov_bool_1_int_3, 'StatsMsg', 76);

proc ark_cov_bool_1_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<bool,1,int64,4>', ark_cov_bool_1_int_4, 'StatsMsg', 76);

proc ark_cov_bool_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<bool,1,uint64,1>', ark_cov_bool_1_uint_1, 'StatsMsg', 76);

proc ark_cov_bool_1_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<bool,1,uint64,2>', ark_cov_bool_1_uint_2, 'StatsMsg', 76);

proc ark_cov_bool_1_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<bool,1,uint64,3>', ark_cov_bool_1_uint_3, 'StatsMsg', 76);

proc ark_cov_bool_1_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<bool,1,uint64,4>', ark_cov_bool_1_uint_4, 'StatsMsg', 76);

proc ark_cov_bool_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<bool,1,uint8,1>', ark_cov_bool_1_uint8_1, 'StatsMsg', 76);

proc ark_cov_bool_1_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<bool,1,uint8,2>', ark_cov_bool_1_uint8_2, 'StatsMsg', 76);

proc ark_cov_bool_1_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<bool,1,uint8,3>', ark_cov_bool_1_uint8_3, 'StatsMsg', 76);

proc ark_cov_bool_1_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<bool,1,uint8,4>', ark_cov_bool_1_uint8_4, 'StatsMsg', 76);

proc ark_cov_bool_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<bool,1,float64,1>', ark_cov_bool_1_real_1, 'StatsMsg', 76);

proc ark_cov_bool_1_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<bool,1,float64,2>', ark_cov_bool_1_real_2, 'StatsMsg', 76);

proc ark_cov_bool_1_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<bool,1,float64,3>', ark_cov_bool_1_real_3, 'StatsMsg', 76);

proc ark_cov_bool_1_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<bool,1,float64,4>', ark_cov_bool_1_real_4, 'StatsMsg', 76);

proc ark_cov_bool_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<bool,1,bool,1>', ark_cov_bool_1_bool_1, 'StatsMsg', 76);

proc ark_cov_bool_1_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<bool,1,bool,2>', ark_cov_bool_1_bool_2, 'StatsMsg', 76);

proc ark_cov_bool_1_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<bool,1,bool,3>', ark_cov_bool_1_bool_3, 'StatsMsg', 76);

proc ark_cov_bool_1_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<bool,1,bool,4>', ark_cov_bool_1_bool_4, 'StatsMsg', 76);

proc ark_cov_bool_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<bool,1,bigint,1>', ark_cov_bool_1_bigint_1, 'StatsMsg', 76);

proc ark_cov_bool_1_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<bool,1,bigint,2>', ark_cov_bool_1_bigint_2, 'StatsMsg', 76);

proc ark_cov_bool_1_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<bool,1,bigint,3>', ark_cov_bool_1_bigint_3, 'StatsMsg', 76);

proc ark_cov_bool_1_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<bool,1,bigint,4>', ark_cov_bool_1_bigint_4, 'StatsMsg', 76);

proc ark_cov_bool_2_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<bool,2,int64,1>', ark_cov_bool_2_int_1, 'StatsMsg', 76);

proc ark_cov_bool_2_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<bool,2,int64,2>', ark_cov_bool_2_int_2, 'StatsMsg', 76);

proc ark_cov_bool_2_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<bool,2,int64,3>', ark_cov_bool_2_int_3, 'StatsMsg', 76);

proc ark_cov_bool_2_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<bool,2,int64,4>', ark_cov_bool_2_int_4, 'StatsMsg', 76);

proc ark_cov_bool_2_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<bool,2,uint64,1>', ark_cov_bool_2_uint_1, 'StatsMsg', 76);

proc ark_cov_bool_2_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<bool,2,uint64,2>', ark_cov_bool_2_uint_2, 'StatsMsg', 76);

proc ark_cov_bool_2_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<bool,2,uint64,3>', ark_cov_bool_2_uint_3, 'StatsMsg', 76);

proc ark_cov_bool_2_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<bool,2,uint64,4>', ark_cov_bool_2_uint_4, 'StatsMsg', 76);

proc ark_cov_bool_2_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<bool,2,uint8,1>', ark_cov_bool_2_uint8_1, 'StatsMsg', 76);

proc ark_cov_bool_2_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<bool,2,uint8,2>', ark_cov_bool_2_uint8_2, 'StatsMsg', 76);

proc ark_cov_bool_2_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<bool,2,uint8,3>', ark_cov_bool_2_uint8_3, 'StatsMsg', 76);

proc ark_cov_bool_2_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<bool,2,uint8,4>', ark_cov_bool_2_uint8_4, 'StatsMsg', 76);

proc ark_cov_bool_2_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<bool,2,float64,1>', ark_cov_bool_2_real_1, 'StatsMsg', 76);

proc ark_cov_bool_2_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<bool,2,float64,2>', ark_cov_bool_2_real_2, 'StatsMsg', 76);

proc ark_cov_bool_2_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<bool,2,float64,3>', ark_cov_bool_2_real_3, 'StatsMsg', 76);

proc ark_cov_bool_2_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<bool,2,float64,4>', ark_cov_bool_2_real_4, 'StatsMsg', 76);

proc ark_cov_bool_2_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<bool,2,bool,1>', ark_cov_bool_2_bool_1, 'StatsMsg', 76);

proc ark_cov_bool_2_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<bool,2,bool,2>', ark_cov_bool_2_bool_2, 'StatsMsg', 76);

proc ark_cov_bool_2_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<bool,2,bool,3>', ark_cov_bool_2_bool_3, 'StatsMsg', 76);

proc ark_cov_bool_2_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<bool,2,bool,4>', ark_cov_bool_2_bool_4, 'StatsMsg', 76);

proc ark_cov_bool_2_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<bool,2,bigint,1>', ark_cov_bool_2_bigint_1, 'StatsMsg', 76);

proc ark_cov_bool_2_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<bool,2,bigint,2>', ark_cov_bool_2_bigint_2, 'StatsMsg', 76);

proc ark_cov_bool_2_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<bool,2,bigint,3>', ark_cov_bool_2_bigint_3, 'StatsMsg', 76);

proc ark_cov_bool_2_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<bool,2,bigint,4>', ark_cov_bool_2_bigint_4, 'StatsMsg', 76);

proc ark_cov_bool_3_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<bool,3,int64,1>', ark_cov_bool_3_int_1, 'StatsMsg', 76);

proc ark_cov_bool_3_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<bool,3,int64,2>', ark_cov_bool_3_int_2, 'StatsMsg', 76);

proc ark_cov_bool_3_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<bool,3,int64,3>', ark_cov_bool_3_int_3, 'StatsMsg', 76);

proc ark_cov_bool_3_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<bool,3,int64,4>', ark_cov_bool_3_int_4, 'StatsMsg', 76);

proc ark_cov_bool_3_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<bool,3,uint64,1>', ark_cov_bool_3_uint_1, 'StatsMsg', 76);

proc ark_cov_bool_3_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<bool,3,uint64,2>', ark_cov_bool_3_uint_2, 'StatsMsg', 76);

proc ark_cov_bool_3_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<bool,3,uint64,3>', ark_cov_bool_3_uint_3, 'StatsMsg', 76);

proc ark_cov_bool_3_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<bool,3,uint64,4>', ark_cov_bool_3_uint_4, 'StatsMsg', 76);

proc ark_cov_bool_3_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<bool,3,uint8,1>', ark_cov_bool_3_uint8_1, 'StatsMsg', 76);

proc ark_cov_bool_3_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<bool,3,uint8,2>', ark_cov_bool_3_uint8_2, 'StatsMsg', 76);

proc ark_cov_bool_3_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<bool,3,uint8,3>', ark_cov_bool_3_uint8_3, 'StatsMsg', 76);

proc ark_cov_bool_3_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<bool,3,uint8,4>', ark_cov_bool_3_uint8_4, 'StatsMsg', 76);

proc ark_cov_bool_3_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<bool,3,float64,1>', ark_cov_bool_3_real_1, 'StatsMsg', 76);

proc ark_cov_bool_3_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<bool,3,float64,2>', ark_cov_bool_3_real_2, 'StatsMsg', 76);

proc ark_cov_bool_3_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<bool,3,float64,3>', ark_cov_bool_3_real_3, 'StatsMsg', 76);

proc ark_cov_bool_3_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<bool,3,float64,4>', ark_cov_bool_3_real_4, 'StatsMsg', 76);

proc ark_cov_bool_3_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<bool,3,bool,1>', ark_cov_bool_3_bool_1, 'StatsMsg', 76);

proc ark_cov_bool_3_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<bool,3,bool,2>', ark_cov_bool_3_bool_2, 'StatsMsg', 76);

proc ark_cov_bool_3_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<bool,3,bool,3>', ark_cov_bool_3_bool_3, 'StatsMsg', 76);

proc ark_cov_bool_3_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<bool,3,bool,4>', ark_cov_bool_3_bool_4, 'StatsMsg', 76);

proc ark_cov_bool_3_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<bool,3,bigint,1>', ark_cov_bool_3_bigint_1, 'StatsMsg', 76);

proc ark_cov_bool_3_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<bool,3,bigint,2>', ark_cov_bool_3_bigint_2, 'StatsMsg', 76);

proc ark_cov_bool_3_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<bool,3,bigint,3>', ark_cov_bool_3_bigint_3, 'StatsMsg', 76);

proc ark_cov_bool_3_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<bool,3,bigint,4>', ark_cov_bool_3_bigint_4, 'StatsMsg', 76);

proc ark_cov_bool_4_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<bool,4,int64,1>', ark_cov_bool_4_int_1, 'StatsMsg', 76);

proc ark_cov_bool_4_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<bool,4,int64,2>', ark_cov_bool_4_int_2, 'StatsMsg', 76);

proc ark_cov_bool_4_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<bool,4,int64,3>', ark_cov_bool_4_int_3, 'StatsMsg', 76);

proc ark_cov_bool_4_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<bool,4,int64,4>', ark_cov_bool_4_int_4, 'StatsMsg', 76);

proc ark_cov_bool_4_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<bool,4,uint64,1>', ark_cov_bool_4_uint_1, 'StatsMsg', 76);

proc ark_cov_bool_4_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<bool,4,uint64,2>', ark_cov_bool_4_uint_2, 'StatsMsg', 76);

proc ark_cov_bool_4_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<bool,4,uint64,3>', ark_cov_bool_4_uint_3, 'StatsMsg', 76);

proc ark_cov_bool_4_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<bool,4,uint64,4>', ark_cov_bool_4_uint_4, 'StatsMsg', 76);

proc ark_cov_bool_4_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<bool,4,uint8,1>', ark_cov_bool_4_uint8_1, 'StatsMsg', 76);

proc ark_cov_bool_4_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<bool,4,uint8,2>', ark_cov_bool_4_uint8_2, 'StatsMsg', 76);

proc ark_cov_bool_4_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<bool,4,uint8,3>', ark_cov_bool_4_uint8_3, 'StatsMsg', 76);

proc ark_cov_bool_4_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<bool,4,uint8,4>', ark_cov_bool_4_uint8_4, 'StatsMsg', 76);

proc ark_cov_bool_4_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<bool,4,float64,1>', ark_cov_bool_4_real_1, 'StatsMsg', 76);

proc ark_cov_bool_4_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<bool,4,float64,2>', ark_cov_bool_4_real_2, 'StatsMsg', 76);

proc ark_cov_bool_4_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<bool,4,float64,3>', ark_cov_bool_4_real_3, 'StatsMsg', 76);

proc ark_cov_bool_4_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<bool,4,float64,4>', ark_cov_bool_4_real_4, 'StatsMsg', 76);

proc ark_cov_bool_4_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<bool,4,bool,1>', ark_cov_bool_4_bool_1, 'StatsMsg', 76);

proc ark_cov_bool_4_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<bool,4,bool,2>', ark_cov_bool_4_bool_2, 'StatsMsg', 76);

proc ark_cov_bool_4_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<bool,4,bool,3>', ark_cov_bool_4_bool_3, 'StatsMsg', 76);

proc ark_cov_bool_4_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<bool,4,bool,4>', ark_cov_bool_4_bool_4, 'StatsMsg', 76);

proc ark_cov_bool_4_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<bool,4,bigint,1>', ark_cov_bool_4_bigint_1, 'StatsMsg', 76);

proc ark_cov_bool_4_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<bool,4,bigint,2>', ark_cov_bool_4_bigint_2, 'StatsMsg', 76);

proc ark_cov_bool_4_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<bool,4,bigint,3>', ark_cov_bool_4_bigint_3, 'StatsMsg', 76);

proc ark_cov_bool_4_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<bool,4,bigint,4>', ark_cov_bool_4_bigint_4, 'StatsMsg', 76);

proc ark_cov_bigint_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<bigint,1,int64,1>', ark_cov_bigint_1_int_1, 'StatsMsg', 76);

proc ark_cov_bigint_1_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<bigint,1,int64,2>', ark_cov_bigint_1_int_2, 'StatsMsg', 76);

proc ark_cov_bigint_1_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<bigint,1,int64,3>', ark_cov_bigint_1_int_3, 'StatsMsg', 76);

proc ark_cov_bigint_1_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<bigint,1,int64,4>', ark_cov_bigint_1_int_4, 'StatsMsg', 76);

proc ark_cov_bigint_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<bigint,1,uint64,1>', ark_cov_bigint_1_uint_1, 'StatsMsg', 76);

proc ark_cov_bigint_1_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<bigint,1,uint64,2>', ark_cov_bigint_1_uint_2, 'StatsMsg', 76);

proc ark_cov_bigint_1_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<bigint,1,uint64,3>', ark_cov_bigint_1_uint_3, 'StatsMsg', 76);

proc ark_cov_bigint_1_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<bigint,1,uint64,4>', ark_cov_bigint_1_uint_4, 'StatsMsg', 76);

proc ark_cov_bigint_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<bigint,1,uint8,1>', ark_cov_bigint_1_uint8_1, 'StatsMsg', 76);

proc ark_cov_bigint_1_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<bigint,1,uint8,2>', ark_cov_bigint_1_uint8_2, 'StatsMsg', 76);

proc ark_cov_bigint_1_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<bigint,1,uint8,3>', ark_cov_bigint_1_uint8_3, 'StatsMsg', 76);

proc ark_cov_bigint_1_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<bigint,1,uint8,4>', ark_cov_bigint_1_uint8_4, 'StatsMsg', 76);

proc ark_cov_bigint_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<bigint,1,float64,1>', ark_cov_bigint_1_real_1, 'StatsMsg', 76);

proc ark_cov_bigint_1_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<bigint,1,float64,2>', ark_cov_bigint_1_real_2, 'StatsMsg', 76);

proc ark_cov_bigint_1_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<bigint,1,float64,3>', ark_cov_bigint_1_real_3, 'StatsMsg', 76);

proc ark_cov_bigint_1_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<bigint,1,float64,4>', ark_cov_bigint_1_real_4, 'StatsMsg', 76);

proc ark_cov_bigint_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<bigint,1,bool,1>', ark_cov_bigint_1_bool_1, 'StatsMsg', 76);

proc ark_cov_bigint_1_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<bigint,1,bool,2>', ark_cov_bigint_1_bool_2, 'StatsMsg', 76);

proc ark_cov_bigint_1_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<bigint,1,bool,3>', ark_cov_bigint_1_bool_3, 'StatsMsg', 76);

proc ark_cov_bigint_1_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<bigint,1,bool,4>', ark_cov_bigint_1_bool_4, 'StatsMsg', 76);

proc ark_cov_bigint_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<bigint,1,bigint,1>', ark_cov_bigint_1_bigint_1, 'StatsMsg', 76);

proc ark_cov_bigint_1_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<bigint,1,bigint,2>', ark_cov_bigint_1_bigint_2, 'StatsMsg', 76);

proc ark_cov_bigint_1_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<bigint,1,bigint,3>', ark_cov_bigint_1_bigint_3, 'StatsMsg', 76);

proc ark_cov_bigint_1_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<bigint,1,bigint,4>', ark_cov_bigint_1_bigint_4, 'StatsMsg', 76);

proc ark_cov_bigint_2_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<bigint,2,int64,1>', ark_cov_bigint_2_int_1, 'StatsMsg', 76);

proc ark_cov_bigint_2_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<bigint,2,int64,2>', ark_cov_bigint_2_int_2, 'StatsMsg', 76);

proc ark_cov_bigint_2_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<bigint,2,int64,3>', ark_cov_bigint_2_int_3, 'StatsMsg', 76);

proc ark_cov_bigint_2_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<bigint,2,int64,4>', ark_cov_bigint_2_int_4, 'StatsMsg', 76);

proc ark_cov_bigint_2_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<bigint,2,uint64,1>', ark_cov_bigint_2_uint_1, 'StatsMsg', 76);

proc ark_cov_bigint_2_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<bigint,2,uint64,2>', ark_cov_bigint_2_uint_2, 'StatsMsg', 76);

proc ark_cov_bigint_2_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<bigint,2,uint64,3>', ark_cov_bigint_2_uint_3, 'StatsMsg', 76);

proc ark_cov_bigint_2_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<bigint,2,uint64,4>', ark_cov_bigint_2_uint_4, 'StatsMsg', 76);

proc ark_cov_bigint_2_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<bigint,2,uint8,1>', ark_cov_bigint_2_uint8_1, 'StatsMsg', 76);

proc ark_cov_bigint_2_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<bigint,2,uint8,2>', ark_cov_bigint_2_uint8_2, 'StatsMsg', 76);

proc ark_cov_bigint_2_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<bigint,2,uint8,3>', ark_cov_bigint_2_uint8_3, 'StatsMsg', 76);

proc ark_cov_bigint_2_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<bigint,2,uint8,4>', ark_cov_bigint_2_uint8_4, 'StatsMsg', 76);

proc ark_cov_bigint_2_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<bigint,2,float64,1>', ark_cov_bigint_2_real_1, 'StatsMsg', 76);

proc ark_cov_bigint_2_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<bigint,2,float64,2>', ark_cov_bigint_2_real_2, 'StatsMsg', 76);

proc ark_cov_bigint_2_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<bigint,2,float64,3>', ark_cov_bigint_2_real_3, 'StatsMsg', 76);

proc ark_cov_bigint_2_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<bigint,2,float64,4>', ark_cov_bigint_2_real_4, 'StatsMsg', 76);

proc ark_cov_bigint_2_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<bigint,2,bool,1>', ark_cov_bigint_2_bool_1, 'StatsMsg', 76);

proc ark_cov_bigint_2_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<bigint,2,bool,2>', ark_cov_bigint_2_bool_2, 'StatsMsg', 76);

proc ark_cov_bigint_2_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<bigint,2,bool,3>', ark_cov_bigint_2_bool_3, 'StatsMsg', 76);

proc ark_cov_bigint_2_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<bigint,2,bool,4>', ark_cov_bigint_2_bool_4, 'StatsMsg', 76);

proc ark_cov_bigint_2_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<bigint,2,bigint,1>', ark_cov_bigint_2_bigint_1, 'StatsMsg', 76);

proc ark_cov_bigint_2_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<bigint,2,bigint,2>', ark_cov_bigint_2_bigint_2, 'StatsMsg', 76);

proc ark_cov_bigint_2_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<bigint,2,bigint,3>', ark_cov_bigint_2_bigint_3, 'StatsMsg', 76);

proc ark_cov_bigint_2_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<bigint,2,bigint,4>', ark_cov_bigint_2_bigint_4, 'StatsMsg', 76);

proc ark_cov_bigint_3_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<bigint,3,int64,1>', ark_cov_bigint_3_int_1, 'StatsMsg', 76);

proc ark_cov_bigint_3_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<bigint,3,int64,2>', ark_cov_bigint_3_int_2, 'StatsMsg', 76);

proc ark_cov_bigint_3_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<bigint,3,int64,3>', ark_cov_bigint_3_int_3, 'StatsMsg', 76);

proc ark_cov_bigint_3_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<bigint,3,int64,4>', ark_cov_bigint_3_int_4, 'StatsMsg', 76);

proc ark_cov_bigint_3_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<bigint,3,uint64,1>', ark_cov_bigint_3_uint_1, 'StatsMsg', 76);

proc ark_cov_bigint_3_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<bigint,3,uint64,2>', ark_cov_bigint_3_uint_2, 'StatsMsg', 76);

proc ark_cov_bigint_3_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<bigint,3,uint64,3>', ark_cov_bigint_3_uint_3, 'StatsMsg', 76);

proc ark_cov_bigint_3_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<bigint,3,uint64,4>', ark_cov_bigint_3_uint_4, 'StatsMsg', 76);

proc ark_cov_bigint_3_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<bigint,3,uint8,1>', ark_cov_bigint_3_uint8_1, 'StatsMsg', 76);

proc ark_cov_bigint_3_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<bigint,3,uint8,2>', ark_cov_bigint_3_uint8_2, 'StatsMsg', 76);

proc ark_cov_bigint_3_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<bigint,3,uint8,3>', ark_cov_bigint_3_uint8_3, 'StatsMsg', 76);

proc ark_cov_bigint_3_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<bigint,3,uint8,4>', ark_cov_bigint_3_uint8_4, 'StatsMsg', 76);

proc ark_cov_bigint_3_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<bigint,3,float64,1>', ark_cov_bigint_3_real_1, 'StatsMsg', 76);

proc ark_cov_bigint_3_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<bigint,3,float64,2>', ark_cov_bigint_3_real_2, 'StatsMsg', 76);

proc ark_cov_bigint_3_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<bigint,3,float64,3>', ark_cov_bigint_3_real_3, 'StatsMsg', 76);

proc ark_cov_bigint_3_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<bigint,3,float64,4>', ark_cov_bigint_3_real_4, 'StatsMsg', 76);

proc ark_cov_bigint_3_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<bigint,3,bool,1>', ark_cov_bigint_3_bool_1, 'StatsMsg', 76);

proc ark_cov_bigint_3_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<bigint,3,bool,2>', ark_cov_bigint_3_bool_2, 'StatsMsg', 76);

proc ark_cov_bigint_3_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<bigint,3,bool,3>', ark_cov_bigint_3_bool_3, 'StatsMsg', 76);

proc ark_cov_bigint_3_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<bigint,3,bool,4>', ark_cov_bigint_3_bool_4, 'StatsMsg', 76);

proc ark_cov_bigint_3_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<bigint,3,bigint,1>', ark_cov_bigint_3_bigint_1, 'StatsMsg', 76);

proc ark_cov_bigint_3_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<bigint,3,bigint,2>', ark_cov_bigint_3_bigint_2, 'StatsMsg', 76);

proc ark_cov_bigint_3_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<bigint,3,bigint,3>', ark_cov_bigint_3_bigint_3, 'StatsMsg', 76);

proc ark_cov_bigint_3_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<bigint,3,bigint,4>', ark_cov_bigint_3_bigint_4, 'StatsMsg', 76);

proc ark_cov_bigint_4_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=int, array_nd_1=1);
registerFunction('cov<bigint,4,int64,1>', ark_cov_bigint_4_int_1, 'StatsMsg', 76);

proc ark_cov_bigint_4_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=int, array_nd_1=2);
registerFunction('cov<bigint,4,int64,2>', ark_cov_bigint_4_int_2, 'StatsMsg', 76);

proc ark_cov_bigint_4_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=int, array_nd_1=3);
registerFunction('cov<bigint,4,int64,3>', ark_cov_bigint_4_int_3, 'StatsMsg', 76);

proc ark_cov_bigint_4_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=int, array_nd_1=4);
registerFunction('cov<bigint,4,int64,4>', ark_cov_bigint_4_int_4, 'StatsMsg', 76);

proc ark_cov_bigint_4_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint, array_nd_1=1);
registerFunction('cov<bigint,4,uint64,1>', ark_cov_bigint_4_uint_1, 'StatsMsg', 76);

proc ark_cov_bigint_4_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint, array_nd_1=2);
registerFunction('cov<bigint,4,uint64,2>', ark_cov_bigint_4_uint_2, 'StatsMsg', 76);

proc ark_cov_bigint_4_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint, array_nd_1=3);
registerFunction('cov<bigint,4,uint64,3>', ark_cov_bigint_4_uint_3, 'StatsMsg', 76);

proc ark_cov_bigint_4_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint, array_nd_1=4);
registerFunction('cov<bigint,4,uint64,4>', ark_cov_bigint_4_uint_4, 'StatsMsg', 76);

proc ark_cov_bigint_4_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('cov<bigint,4,uint8,1>', ark_cov_bigint_4_uint8_1, 'StatsMsg', 76);

proc ark_cov_bigint_4_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('cov<bigint,4,uint8,2>', ark_cov_bigint_4_uint8_2, 'StatsMsg', 76);

proc ark_cov_bigint_4_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('cov<bigint,4,uint8,3>', ark_cov_bigint_4_uint8_3, 'StatsMsg', 76);

proc ark_cov_bigint_4_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('cov<bigint,4,uint8,4>', ark_cov_bigint_4_uint8_4, 'StatsMsg', 76);

proc ark_cov_bigint_4_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=real, array_nd_1=1);
registerFunction('cov<bigint,4,float64,1>', ark_cov_bigint_4_real_1, 'StatsMsg', 76);

proc ark_cov_bigint_4_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=real, array_nd_1=2);
registerFunction('cov<bigint,4,float64,2>', ark_cov_bigint_4_real_2, 'StatsMsg', 76);

proc ark_cov_bigint_4_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=real, array_nd_1=3);
registerFunction('cov<bigint,4,float64,3>', ark_cov_bigint_4_real_3, 'StatsMsg', 76);

proc ark_cov_bigint_4_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=real, array_nd_1=4);
registerFunction('cov<bigint,4,float64,4>', ark_cov_bigint_4_real_4, 'StatsMsg', 76);

proc ark_cov_bigint_4_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bool, array_nd_1=1);
registerFunction('cov<bigint,4,bool,1>', ark_cov_bigint_4_bool_1, 'StatsMsg', 76);

proc ark_cov_bigint_4_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bool, array_nd_1=2);
registerFunction('cov<bigint,4,bool,2>', ark_cov_bigint_4_bool_2, 'StatsMsg', 76);

proc ark_cov_bigint_4_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bool, array_nd_1=3);
registerFunction('cov<bigint,4,bool,3>', ark_cov_bigint_4_bool_3, 'StatsMsg', 76);

proc ark_cov_bigint_4_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bool, array_nd_1=4);
registerFunction('cov<bigint,4,bool,4>', ark_cov_bigint_4_bool_4, 'StatsMsg', 76);

proc ark_cov_bigint_4_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=1);
registerFunction('cov<bigint,4,bigint,1>', ark_cov_bigint_4_bigint_1, 'StatsMsg', 76);

proc ark_cov_bigint_4_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=2);
registerFunction('cov<bigint,4,bigint,2>', ark_cov_bigint_4_bigint_2, 'StatsMsg', 76);

proc ark_cov_bigint_4_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=3);
registerFunction('cov<bigint,4,bigint,3>', ark_cov_bigint_4_bigint_3, 'StatsMsg', 76);

proc ark_cov_bigint_4_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cov_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=4);
registerFunction('cov<bigint,4,bigint,4>', ark_cov_bigint_4_bigint_4, 'StatsMsg', 76);

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

proc ark_corr_int_1_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<int64,1,int64,2>', ark_corr_int_1_int_2, 'StatsMsg', 98);

proc ark_corr_int_1_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<int64,1,int64,3>', ark_corr_int_1_int_3, 'StatsMsg', 98);

proc ark_corr_int_1_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<int64,1,int64,4>', ark_corr_int_1_int_4, 'StatsMsg', 98);

proc ark_corr_int_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<int64,1,uint64,1>', ark_corr_int_1_uint_1, 'StatsMsg', 98);

proc ark_corr_int_1_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<int64,1,uint64,2>', ark_corr_int_1_uint_2, 'StatsMsg', 98);

proc ark_corr_int_1_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<int64,1,uint64,3>', ark_corr_int_1_uint_3, 'StatsMsg', 98);

proc ark_corr_int_1_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<int64,1,uint64,4>', ark_corr_int_1_uint_4, 'StatsMsg', 98);

proc ark_corr_int_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<int64,1,uint8,1>', ark_corr_int_1_uint8_1, 'StatsMsg', 98);

proc ark_corr_int_1_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<int64,1,uint8,2>', ark_corr_int_1_uint8_2, 'StatsMsg', 98);

proc ark_corr_int_1_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<int64,1,uint8,3>', ark_corr_int_1_uint8_3, 'StatsMsg', 98);

proc ark_corr_int_1_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<int64,1,uint8,4>', ark_corr_int_1_uint8_4, 'StatsMsg', 98);

proc ark_corr_int_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<int64,1,float64,1>', ark_corr_int_1_real_1, 'StatsMsg', 98);

proc ark_corr_int_1_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<int64,1,float64,2>', ark_corr_int_1_real_2, 'StatsMsg', 98);

proc ark_corr_int_1_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<int64,1,float64,3>', ark_corr_int_1_real_3, 'StatsMsg', 98);

proc ark_corr_int_1_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<int64,1,float64,4>', ark_corr_int_1_real_4, 'StatsMsg', 98);

proc ark_corr_int_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<int64,1,bool,1>', ark_corr_int_1_bool_1, 'StatsMsg', 98);

proc ark_corr_int_1_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<int64,1,bool,2>', ark_corr_int_1_bool_2, 'StatsMsg', 98);

proc ark_corr_int_1_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<int64,1,bool,3>', ark_corr_int_1_bool_3, 'StatsMsg', 98);

proc ark_corr_int_1_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<int64,1,bool,4>', ark_corr_int_1_bool_4, 'StatsMsg', 98);

proc ark_corr_int_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<int64,1,bigint,1>', ark_corr_int_1_bigint_1, 'StatsMsg', 98);

proc ark_corr_int_1_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<int64,1,bigint,2>', ark_corr_int_1_bigint_2, 'StatsMsg', 98);

proc ark_corr_int_1_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<int64,1,bigint,3>', ark_corr_int_1_bigint_3, 'StatsMsg', 98);

proc ark_corr_int_1_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<int64,1,bigint,4>', ark_corr_int_1_bigint_4, 'StatsMsg', 98);

proc ark_corr_int_2_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<int64,2,int64,1>', ark_corr_int_2_int_1, 'StatsMsg', 98);

proc ark_corr_int_2_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<int64,2,int64,2>', ark_corr_int_2_int_2, 'StatsMsg', 98);

proc ark_corr_int_2_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<int64,2,int64,3>', ark_corr_int_2_int_3, 'StatsMsg', 98);

proc ark_corr_int_2_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<int64,2,int64,4>', ark_corr_int_2_int_4, 'StatsMsg', 98);

proc ark_corr_int_2_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<int64,2,uint64,1>', ark_corr_int_2_uint_1, 'StatsMsg', 98);

proc ark_corr_int_2_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<int64,2,uint64,2>', ark_corr_int_2_uint_2, 'StatsMsg', 98);

proc ark_corr_int_2_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<int64,2,uint64,3>', ark_corr_int_2_uint_3, 'StatsMsg', 98);

proc ark_corr_int_2_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<int64,2,uint64,4>', ark_corr_int_2_uint_4, 'StatsMsg', 98);

proc ark_corr_int_2_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<int64,2,uint8,1>', ark_corr_int_2_uint8_1, 'StatsMsg', 98);

proc ark_corr_int_2_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<int64,2,uint8,2>', ark_corr_int_2_uint8_2, 'StatsMsg', 98);

proc ark_corr_int_2_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<int64,2,uint8,3>', ark_corr_int_2_uint8_3, 'StatsMsg', 98);

proc ark_corr_int_2_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<int64,2,uint8,4>', ark_corr_int_2_uint8_4, 'StatsMsg', 98);

proc ark_corr_int_2_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<int64,2,float64,1>', ark_corr_int_2_real_1, 'StatsMsg', 98);

proc ark_corr_int_2_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<int64,2,float64,2>', ark_corr_int_2_real_2, 'StatsMsg', 98);

proc ark_corr_int_2_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<int64,2,float64,3>', ark_corr_int_2_real_3, 'StatsMsg', 98);

proc ark_corr_int_2_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<int64,2,float64,4>', ark_corr_int_2_real_4, 'StatsMsg', 98);

proc ark_corr_int_2_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<int64,2,bool,1>', ark_corr_int_2_bool_1, 'StatsMsg', 98);

proc ark_corr_int_2_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<int64,2,bool,2>', ark_corr_int_2_bool_2, 'StatsMsg', 98);

proc ark_corr_int_2_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<int64,2,bool,3>', ark_corr_int_2_bool_3, 'StatsMsg', 98);

proc ark_corr_int_2_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<int64,2,bool,4>', ark_corr_int_2_bool_4, 'StatsMsg', 98);

proc ark_corr_int_2_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<int64,2,bigint,1>', ark_corr_int_2_bigint_1, 'StatsMsg', 98);

proc ark_corr_int_2_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<int64,2,bigint,2>', ark_corr_int_2_bigint_2, 'StatsMsg', 98);

proc ark_corr_int_2_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<int64,2,bigint,3>', ark_corr_int_2_bigint_3, 'StatsMsg', 98);

proc ark_corr_int_2_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<int64,2,bigint,4>', ark_corr_int_2_bigint_4, 'StatsMsg', 98);

proc ark_corr_int_3_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<int64,3,int64,1>', ark_corr_int_3_int_1, 'StatsMsg', 98);

proc ark_corr_int_3_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<int64,3,int64,2>', ark_corr_int_3_int_2, 'StatsMsg', 98);

proc ark_corr_int_3_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<int64,3,int64,3>', ark_corr_int_3_int_3, 'StatsMsg', 98);

proc ark_corr_int_3_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<int64,3,int64,4>', ark_corr_int_3_int_4, 'StatsMsg', 98);

proc ark_corr_int_3_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<int64,3,uint64,1>', ark_corr_int_3_uint_1, 'StatsMsg', 98);

proc ark_corr_int_3_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<int64,3,uint64,2>', ark_corr_int_3_uint_2, 'StatsMsg', 98);

proc ark_corr_int_3_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<int64,3,uint64,3>', ark_corr_int_3_uint_3, 'StatsMsg', 98);

proc ark_corr_int_3_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<int64,3,uint64,4>', ark_corr_int_3_uint_4, 'StatsMsg', 98);

proc ark_corr_int_3_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<int64,3,uint8,1>', ark_corr_int_3_uint8_1, 'StatsMsg', 98);

proc ark_corr_int_3_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<int64,3,uint8,2>', ark_corr_int_3_uint8_2, 'StatsMsg', 98);

proc ark_corr_int_3_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<int64,3,uint8,3>', ark_corr_int_3_uint8_3, 'StatsMsg', 98);

proc ark_corr_int_3_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<int64,3,uint8,4>', ark_corr_int_3_uint8_4, 'StatsMsg', 98);

proc ark_corr_int_3_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<int64,3,float64,1>', ark_corr_int_3_real_1, 'StatsMsg', 98);

proc ark_corr_int_3_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<int64,3,float64,2>', ark_corr_int_3_real_2, 'StatsMsg', 98);

proc ark_corr_int_3_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<int64,3,float64,3>', ark_corr_int_3_real_3, 'StatsMsg', 98);

proc ark_corr_int_3_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<int64,3,float64,4>', ark_corr_int_3_real_4, 'StatsMsg', 98);

proc ark_corr_int_3_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<int64,3,bool,1>', ark_corr_int_3_bool_1, 'StatsMsg', 98);

proc ark_corr_int_3_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<int64,3,bool,2>', ark_corr_int_3_bool_2, 'StatsMsg', 98);

proc ark_corr_int_3_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<int64,3,bool,3>', ark_corr_int_3_bool_3, 'StatsMsg', 98);

proc ark_corr_int_3_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<int64,3,bool,4>', ark_corr_int_3_bool_4, 'StatsMsg', 98);

proc ark_corr_int_3_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<int64,3,bigint,1>', ark_corr_int_3_bigint_1, 'StatsMsg', 98);

proc ark_corr_int_3_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<int64,3,bigint,2>', ark_corr_int_3_bigint_2, 'StatsMsg', 98);

proc ark_corr_int_3_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<int64,3,bigint,3>', ark_corr_int_3_bigint_3, 'StatsMsg', 98);

proc ark_corr_int_3_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<int64,3,bigint,4>', ark_corr_int_3_bigint_4, 'StatsMsg', 98);

proc ark_corr_int_4_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<int64,4,int64,1>', ark_corr_int_4_int_1, 'StatsMsg', 98);

proc ark_corr_int_4_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<int64,4,int64,2>', ark_corr_int_4_int_2, 'StatsMsg', 98);

proc ark_corr_int_4_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<int64,4,int64,3>', ark_corr_int_4_int_3, 'StatsMsg', 98);

proc ark_corr_int_4_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<int64,4,int64,4>', ark_corr_int_4_int_4, 'StatsMsg', 98);

proc ark_corr_int_4_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<int64,4,uint64,1>', ark_corr_int_4_uint_1, 'StatsMsg', 98);

proc ark_corr_int_4_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<int64,4,uint64,2>', ark_corr_int_4_uint_2, 'StatsMsg', 98);

proc ark_corr_int_4_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<int64,4,uint64,3>', ark_corr_int_4_uint_3, 'StatsMsg', 98);

proc ark_corr_int_4_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<int64,4,uint64,4>', ark_corr_int_4_uint_4, 'StatsMsg', 98);

proc ark_corr_int_4_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<int64,4,uint8,1>', ark_corr_int_4_uint8_1, 'StatsMsg', 98);

proc ark_corr_int_4_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<int64,4,uint8,2>', ark_corr_int_4_uint8_2, 'StatsMsg', 98);

proc ark_corr_int_4_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<int64,4,uint8,3>', ark_corr_int_4_uint8_3, 'StatsMsg', 98);

proc ark_corr_int_4_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<int64,4,uint8,4>', ark_corr_int_4_uint8_4, 'StatsMsg', 98);

proc ark_corr_int_4_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<int64,4,float64,1>', ark_corr_int_4_real_1, 'StatsMsg', 98);

proc ark_corr_int_4_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<int64,4,float64,2>', ark_corr_int_4_real_2, 'StatsMsg', 98);

proc ark_corr_int_4_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<int64,4,float64,3>', ark_corr_int_4_real_3, 'StatsMsg', 98);

proc ark_corr_int_4_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<int64,4,float64,4>', ark_corr_int_4_real_4, 'StatsMsg', 98);

proc ark_corr_int_4_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<int64,4,bool,1>', ark_corr_int_4_bool_1, 'StatsMsg', 98);

proc ark_corr_int_4_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<int64,4,bool,2>', ark_corr_int_4_bool_2, 'StatsMsg', 98);

proc ark_corr_int_4_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<int64,4,bool,3>', ark_corr_int_4_bool_3, 'StatsMsg', 98);

proc ark_corr_int_4_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<int64,4,bool,4>', ark_corr_int_4_bool_4, 'StatsMsg', 98);

proc ark_corr_int_4_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<int64,4,bigint,1>', ark_corr_int_4_bigint_1, 'StatsMsg', 98);

proc ark_corr_int_4_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<int64,4,bigint,2>', ark_corr_int_4_bigint_2, 'StatsMsg', 98);

proc ark_corr_int_4_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<int64,4,bigint,3>', ark_corr_int_4_bigint_3, 'StatsMsg', 98);

proc ark_corr_int_4_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<int64,4,bigint,4>', ark_corr_int_4_bigint_4, 'StatsMsg', 98);

proc ark_corr_uint_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<uint64,1,int64,1>', ark_corr_uint_1_int_1, 'StatsMsg', 98);

proc ark_corr_uint_1_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<uint64,1,int64,2>', ark_corr_uint_1_int_2, 'StatsMsg', 98);

proc ark_corr_uint_1_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<uint64,1,int64,3>', ark_corr_uint_1_int_3, 'StatsMsg', 98);

proc ark_corr_uint_1_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<uint64,1,int64,4>', ark_corr_uint_1_int_4, 'StatsMsg', 98);

proc ark_corr_uint_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<uint64,1,uint64,1>', ark_corr_uint_1_uint_1, 'StatsMsg', 98);

proc ark_corr_uint_1_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<uint64,1,uint64,2>', ark_corr_uint_1_uint_2, 'StatsMsg', 98);

proc ark_corr_uint_1_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<uint64,1,uint64,3>', ark_corr_uint_1_uint_3, 'StatsMsg', 98);

proc ark_corr_uint_1_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<uint64,1,uint64,4>', ark_corr_uint_1_uint_4, 'StatsMsg', 98);

proc ark_corr_uint_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<uint64,1,uint8,1>', ark_corr_uint_1_uint8_1, 'StatsMsg', 98);

proc ark_corr_uint_1_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<uint64,1,uint8,2>', ark_corr_uint_1_uint8_2, 'StatsMsg', 98);

proc ark_corr_uint_1_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<uint64,1,uint8,3>', ark_corr_uint_1_uint8_3, 'StatsMsg', 98);

proc ark_corr_uint_1_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<uint64,1,uint8,4>', ark_corr_uint_1_uint8_4, 'StatsMsg', 98);

proc ark_corr_uint_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<uint64,1,float64,1>', ark_corr_uint_1_real_1, 'StatsMsg', 98);

proc ark_corr_uint_1_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<uint64,1,float64,2>', ark_corr_uint_1_real_2, 'StatsMsg', 98);

proc ark_corr_uint_1_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<uint64,1,float64,3>', ark_corr_uint_1_real_3, 'StatsMsg', 98);

proc ark_corr_uint_1_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<uint64,1,float64,4>', ark_corr_uint_1_real_4, 'StatsMsg', 98);

proc ark_corr_uint_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<uint64,1,bool,1>', ark_corr_uint_1_bool_1, 'StatsMsg', 98);

proc ark_corr_uint_1_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<uint64,1,bool,2>', ark_corr_uint_1_bool_2, 'StatsMsg', 98);

proc ark_corr_uint_1_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<uint64,1,bool,3>', ark_corr_uint_1_bool_3, 'StatsMsg', 98);

proc ark_corr_uint_1_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<uint64,1,bool,4>', ark_corr_uint_1_bool_4, 'StatsMsg', 98);

proc ark_corr_uint_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<uint64,1,bigint,1>', ark_corr_uint_1_bigint_1, 'StatsMsg', 98);

proc ark_corr_uint_1_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<uint64,1,bigint,2>', ark_corr_uint_1_bigint_2, 'StatsMsg', 98);

proc ark_corr_uint_1_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<uint64,1,bigint,3>', ark_corr_uint_1_bigint_3, 'StatsMsg', 98);

proc ark_corr_uint_1_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<uint64,1,bigint,4>', ark_corr_uint_1_bigint_4, 'StatsMsg', 98);

proc ark_corr_uint_2_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<uint64,2,int64,1>', ark_corr_uint_2_int_1, 'StatsMsg', 98);

proc ark_corr_uint_2_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<uint64,2,int64,2>', ark_corr_uint_2_int_2, 'StatsMsg', 98);

proc ark_corr_uint_2_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<uint64,2,int64,3>', ark_corr_uint_2_int_3, 'StatsMsg', 98);

proc ark_corr_uint_2_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<uint64,2,int64,4>', ark_corr_uint_2_int_4, 'StatsMsg', 98);

proc ark_corr_uint_2_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<uint64,2,uint64,1>', ark_corr_uint_2_uint_1, 'StatsMsg', 98);

proc ark_corr_uint_2_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<uint64,2,uint64,2>', ark_corr_uint_2_uint_2, 'StatsMsg', 98);

proc ark_corr_uint_2_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<uint64,2,uint64,3>', ark_corr_uint_2_uint_3, 'StatsMsg', 98);

proc ark_corr_uint_2_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<uint64,2,uint64,4>', ark_corr_uint_2_uint_4, 'StatsMsg', 98);

proc ark_corr_uint_2_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<uint64,2,uint8,1>', ark_corr_uint_2_uint8_1, 'StatsMsg', 98);

proc ark_corr_uint_2_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<uint64,2,uint8,2>', ark_corr_uint_2_uint8_2, 'StatsMsg', 98);

proc ark_corr_uint_2_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<uint64,2,uint8,3>', ark_corr_uint_2_uint8_3, 'StatsMsg', 98);

proc ark_corr_uint_2_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<uint64,2,uint8,4>', ark_corr_uint_2_uint8_4, 'StatsMsg', 98);

proc ark_corr_uint_2_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<uint64,2,float64,1>', ark_corr_uint_2_real_1, 'StatsMsg', 98);

proc ark_corr_uint_2_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<uint64,2,float64,2>', ark_corr_uint_2_real_2, 'StatsMsg', 98);

proc ark_corr_uint_2_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<uint64,2,float64,3>', ark_corr_uint_2_real_3, 'StatsMsg', 98);

proc ark_corr_uint_2_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<uint64,2,float64,4>', ark_corr_uint_2_real_4, 'StatsMsg', 98);

proc ark_corr_uint_2_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<uint64,2,bool,1>', ark_corr_uint_2_bool_1, 'StatsMsg', 98);

proc ark_corr_uint_2_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<uint64,2,bool,2>', ark_corr_uint_2_bool_2, 'StatsMsg', 98);

proc ark_corr_uint_2_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<uint64,2,bool,3>', ark_corr_uint_2_bool_3, 'StatsMsg', 98);

proc ark_corr_uint_2_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<uint64,2,bool,4>', ark_corr_uint_2_bool_4, 'StatsMsg', 98);

proc ark_corr_uint_2_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<uint64,2,bigint,1>', ark_corr_uint_2_bigint_1, 'StatsMsg', 98);

proc ark_corr_uint_2_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<uint64,2,bigint,2>', ark_corr_uint_2_bigint_2, 'StatsMsg', 98);

proc ark_corr_uint_2_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<uint64,2,bigint,3>', ark_corr_uint_2_bigint_3, 'StatsMsg', 98);

proc ark_corr_uint_2_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<uint64,2,bigint,4>', ark_corr_uint_2_bigint_4, 'StatsMsg', 98);

proc ark_corr_uint_3_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<uint64,3,int64,1>', ark_corr_uint_3_int_1, 'StatsMsg', 98);

proc ark_corr_uint_3_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<uint64,3,int64,2>', ark_corr_uint_3_int_2, 'StatsMsg', 98);

proc ark_corr_uint_3_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<uint64,3,int64,3>', ark_corr_uint_3_int_3, 'StatsMsg', 98);

proc ark_corr_uint_3_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<uint64,3,int64,4>', ark_corr_uint_3_int_4, 'StatsMsg', 98);

proc ark_corr_uint_3_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<uint64,3,uint64,1>', ark_corr_uint_3_uint_1, 'StatsMsg', 98);

proc ark_corr_uint_3_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<uint64,3,uint64,2>', ark_corr_uint_3_uint_2, 'StatsMsg', 98);

proc ark_corr_uint_3_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<uint64,3,uint64,3>', ark_corr_uint_3_uint_3, 'StatsMsg', 98);

proc ark_corr_uint_3_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<uint64,3,uint64,4>', ark_corr_uint_3_uint_4, 'StatsMsg', 98);

proc ark_corr_uint_3_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<uint64,3,uint8,1>', ark_corr_uint_3_uint8_1, 'StatsMsg', 98);

proc ark_corr_uint_3_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<uint64,3,uint8,2>', ark_corr_uint_3_uint8_2, 'StatsMsg', 98);

proc ark_corr_uint_3_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<uint64,3,uint8,3>', ark_corr_uint_3_uint8_3, 'StatsMsg', 98);

proc ark_corr_uint_3_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<uint64,3,uint8,4>', ark_corr_uint_3_uint8_4, 'StatsMsg', 98);

proc ark_corr_uint_3_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<uint64,3,float64,1>', ark_corr_uint_3_real_1, 'StatsMsg', 98);

proc ark_corr_uint_3_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<uint64,3,float64,2>', ark_corr_uint_3_real_2, 'StatsMsg', 98);

proc ark_corr_uint_3_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<uint64,3,float64,3>', ark_corr_uint_3_real_3, 'StatsMsg', 98);

proc ark_corr_uint_3_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<uint64,3,float64,4>', ark_corr_uint_3_real_4, 'StatsMsg', 98);

proc ark_corr_uint_3_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<uint64,3,bool,1>', ark_corr_uint_3_bool_1, 'StatsMsg', 98);

proc ark_corr_uint_3_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<uint64,3,bool,2>', ark_corr_uint_3_bool_2, 'StatsMsg', 98);

proc ark_corr_uint_3_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<uint64,3,bool,3>', ark_corr_uint_3_bool_3, 'StatsMsg', 98);

proc ark_corr_uint_3_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<uint64,3,bool,4>', ark_corr_uint_3_bool_4, 'StatsMsg', 98);

proc ark_corr_uint_3_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<uint64,3,bigint,1>', ark_corr_uint_3_bigint_1, 'StatsMsg', 98);

proc ark_corr_uint_3_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<uint64,3,bigint,2>', ark_corr_uint_3_bigint_2, 'StatsMsg', 98);

proc ark_corr_uint_3_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<uint64,3,bigint,3>', ark_corr_uint_3_bigint_3, 'StatsMsg', 98);

proc ark_corr_uint_3_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<uint64,3,bigint,4>', ark_corr_uint_3_bigint_4, 'StatsMsg', 98);

proc ark_corr_uint_4_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<uint64,4,int64,1>', ark_corr_uint_4_int_1, 'StatsMsg', 98);

proc ark_corr_uint_4_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<uint64,4,int64,2>', ark_corr_uint_4_int_2, 'StatsMsg', 98);

proc ark_corr_uint_4_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<uint64,4,int64,3>', ark_corr_uint_4_int_3, 'StatsMsg', 98);

proc ark_corr_uint_4_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<uint64,4,int64,4>', ark_corr_uint_4_int_4, 'StatsMsg', 98);

proc ark_corr_uint_4_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<uint64,4,uint64,1>', ark_corr_uint_4_uint_1, 'StatsMsg', 98);

proc ark_corr_uint_4_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<uint64,4,uint64,2>', ark_corr_uint_4_uint_2, 'StatsMsg', 98);

proc ark_corr_uint_4_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<uint64,4,uint64,3>', ark_corr_uint_4_uint_3, 'StatsMsg', 98);

proc ark_corr_uint_4_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<uint64,4,uint64,4>', ark_corr_uint_4_uint_4, 'StatsMsg', 98);

proc ark_corr_uint_4_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<uint64,4,uint8,1>', ark_corr_uint_4_uint8_1, 'StatsMsg', 98);

proc ark_corr_uint_4_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<uint64,4,uint8,2>', ark_corr_uint_4_uint8_2, 'StatsMsg', 98);

proc ark_corr_uint_4_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<uint64,4,uint8,3>', ark_corr_uint_4_uint8_3, 'StatsMsg', 98);

proc ark_corr_uint_4_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<uint64,4,uint8,4>', ark_corr_uint_4_uint8_4, 'StatsMsg', 98);

proc ark_corr_uint_4_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<uint64,4,float64,1>', ark_corr_uint_4_real_1, 'StatsMsg', 98);

proc ark_corr_uint_4_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<uint64,4,float64,2>', ark_corr_uint_4_real_2, 'StatsMsg', 98);

proc ark_corr_uint_4_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<uint64,4,float64,3>', ark_corr_uint_4_real_3, 'StatsMsg', 98);

proc ark_corr_uint_4_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<uint64,4,float64,4>', ark_corr_uint_4_real_4, 'StatsMsg', 98);

proc ark_corr_uint_4_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<uint64,4,bool,1>', ark_corr_uint_4_bool_1, 'StatsMsg', 98);

proc ark_corr_uint_4_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<uint64,4,bool,2>', ark_corr_uint_4_bool_2, 'StatsMsg', 98);

proc ark_corr_uint_4_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<uint64,4,bool,3>', ark_corr_uint_4_bool_3, 'StatsMsg', 98);

proc ark_corr_uint_4_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<uint64,4,bool,4>', ark_corr_uint_4_bool_4, 'StatsMsg', 98);

proc ark_corr_uint_4_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<uint64,4,bigint,1>', ark_corr_uint_4_bigint_1, 'StatsMsg', 98);

proc ark_corr_uint_4_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<uint64,4,bigint,2>', ark_corr_uint_4_bigint_2, 'StatsMsg', 98);

proc ark_corr_uint_4_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<uint64,4,bigint,3>', ark_corr_uint_4_bigint_3, 'StatsMsg', 98);

proc ark_corr_uint_4_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<uint64,4,bigint,4>', ark_corr_uint_4_bigint_4, 'StatsMsg', 98);

proc ark_corr_uint8_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<uint8,1,int64,1>', ark_corr_uint8_1_int_1, 'StatsMsg', 98);

proc ark_corr_uint8_1_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<uint8,1,int64,2>', ark_corr_uint8_1_int_2, 'StatsMsg', 98);

proc ark_corr_uint8_1_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<uint8,1,int64,3>', ark_corr_uint8_1_int_3, 'StatsMsg', 98);

proc ark_corr_uint8_1_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<uint8,1,int64,4>', ark_corr_uint8_1_int_4, 'StatsMsg', 98);

proc ark_corr_uint8_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<uint8,1,uint64,1>', ark_corr_uint8_1_uint_1, 'StatsMsg', 98);

proc ark_corr_uint8_1_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<uint8,1,uint64,2>', ark_corr_uint8_1_uint_2, 'StatsMsg', 98);

proc ark_corr_uint8_1_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<uint8,1,uint64,3>', ark_corr_uint8_1_uint_3, 'StatsMsg', 98);

proc ark_corr_uint8_1_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<uint8,1,uint64,4>', ark_corr_uint8_1_uint_4, 'StatsMsg', 98);

proc ark_corr_uint8_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<uint8,1,uint8,1>', ark_corr_uint8_1_uint8_1, 'StatsMsg', 98);

proc ark_corr_uint8_1_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<uint8,1,uint8,2>', ark_corr_uint8_1_uint8_2, 'StatsMsg', 98);

proc ark_corr_uint8_1_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<uint8,1,uint8,3>', ark_corr_uint8_1_uint8_3, 'StatsMsg', 98);

proc ark_corr_uint8_1_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<uint8,1,uint8,4>', ark_corr_uint8_1_uint8_4, 'StatsMsg', 98);

proc ark_corr_uint8_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<uint8,1,float64,1>', ark_corr_uint8_1_real_1, 'StatsMsg', 98);

proc ark_corr_uint8_1_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<uint8,1,float64,2>', ark_corr_uint8_1_real_2, 'StatsMsg', 98);

proc ark_corr_uint8_1_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<uint8,1,float64,3>', ark_corr_uint8_1_real_3, 'StatsMsg', 98);

proc ark_corr_uint8_1_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<uint8,1,float64,4>', ark_corr_uint8_1_real_4, 'StatsMsg', 98);

proc ark_corr_uint8_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<uint8,1,bool,1>', ark_corr_uint8_1_bool_1, 'StatsMsg', 98);

proc ark_corr_uint8_1_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<uint8,1,bool,2>', ark_corr_uint8_1_bool_2, 'StatsMsg', 98);

proc ark_corr_uint8_1_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<uint8,1,bool,3>', ark_corr_uint8_1_bool_3, 'StatsMsg', 98);

proc ark_corr_uint8_1_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<uint8,1,bool,4>', ark_corr_uint8_1_bool_4, 'StatsMsg', 98);

proc ark_corr_uint8_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<uint8,1,bigint,1>', ark_corr_uint8_1_bigint_1, 'StatsMsg', 98);

proc ark_corr_uint8_1_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<uint8,1,bigint,2>', ark_corr_uint8_1_bigint_2, 'StatsMsg', 98);

proc ark_corr_uint8_1_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<uint8,1,bigint,3>', ark_corr_uint8_1_bigint_3, 'StatsMsg', 98);

proc ark_corr_uint8_1_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<uint8,1,bigint,4>', ark_corr_uint8_1_bigint_4, 'StatsMsg', 98);

proc ark_corr_uint8_2_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<uint8,2,int64,1>', ark_corr_uint8_2_int_1, 'StatsMsg', 98);

proc ark_corr_uint8_2_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<uint8,2,int64,2>', ark_corr_uint8_2_int_2, 'StatsMsg', 98);

proc ark_corr_uint8_2_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<uint8,2,int64,3>', ark_corr_uint8_2_int_3, 'StatsMsg', 98);

proc ark_corr_uint8_2_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<uint8,2,int64,4>', ark_corr_uint8_2_int_4, 'StatsMsg', 98);

proc ark_corr_uint8_2_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<uint8,2,uint64,1>', ark_corr_uint8_2_uint_1, 'StatsMsg', 98);

proc ark_corr_uint8_2_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<uint8,2,uint64,2>', ark_corr_uint8_2_uint_2, 'StatsMsg', 98);

proc ark_corr_uint8_2_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<uint8,2,uint64,3>', ark_corr_uint8_2_uint_3, 'StatsMsg', 98);

proc ark_corr_uint8_2_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<uint8,2,uint64,4>', ark_corr_uint8_2_uint_4, 'StatsMsg', 98);

proc ark_corr_uint8_2_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<uint8,2,uint8,1>', ark_corr_uint8_2_uint8_1, 'StatsMsg', 98);

proc ark_corr_uint8_2_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<uint8,2,uint8,2>', ark_corr_uint8_2_uint8_2, 'StatsMsg', 98);

proc ark_corr_uint8_2_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<uint8,2,uint8,3>', ark_corr_uint8_2_uint8_3, 'StatsMsg', 98);

proc ark_corr_uint8_2_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<uint8,2,uint8,4>', ark_corr_uint8_2_uint8_4, 'StatsMsg', 98);

proc ark_corr_uint8_2_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<uint8,2,float64,1>', ark_corr_uint8_2_real_1, 'StatsMsg', 98);

proc ark_corr_uint8_2_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<uint8,2,float64,2>', ark_corr_uint8_2_real_2, 'StatsMsg', 98);

proc ark_corr_uint8_2_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<uint8,2,float64,3>', ark_corr_uint8_2_real_3, 'StatsMsg', 98);

proc ark_corr_uint8_2_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<uint8,2,float64,4>', ark_corr_uint8_2_real_4, 'StatsMsg', 98);

proc ark_corr_uint8_2_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<uint8,2,bool,1>', ark_corr_uint8_2_bool_1, 'StatsMsg', 98);

proc ark_corr_uint8_2_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<uint8,2,bool,2>', ark_corr_uint8_2_bool_2, 'StatsMsg', 98);

proc ark_corr_uint8_2_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<uint8,2,bool,3>', ark_corr_uint8_2_bool_3, 'StatsMsg', 98);

proc ark_corr_uint8_2_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<uint8,2,bool,4>', ark_corr_uint8_2_bool_4, 'StatsMsg', 98);

proc ark_corr_uint8_2_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<uint8,2,bigint,1>', ark_corr_uint8_2_bigint_1, 'StatsMsg', 98);

proc ark_corr_uint8_2_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<uint8,2,bigint,2>', ark_corr_uint8_2_bigint_2, 'StatsMsg', 98);

proc ark_corr_uint8_2_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<uint8,2,bigint,3>', ark_corr_uint8_2_bigint_3, 'StatsMsg', 98);

proc ark_corr_uint8_2_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<uint8,2,bigint,4>', ark_corr_uint8_2_bigint_4, 'StatsMsg', 98);

proc ark_corr_uint8_3_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<uint8,3,int64,1>', ark_corr_uint8_3_int_1, 'StatsMsg', 98);

proc ark_corr_uint8_3_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<uint8,3,int64,2>', ark_corr_uint8_3_int_2, 'StatsMsg', 98);

proc ark_corr_uint8_3_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<uint8,3,int64,3>', ark_corr_uint8_3_int_3, 'StatsMsg', 98);

proc ark_corr_uint8_3_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<uint8,3,int64,4>', ark_corr_uint8_3_int_4, 'StatsMsg', 98);

proc ark_corr_uint8_3_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<uint8,3,uint64,1>', ark_corr_uint8_3_uint_1, 'StatsMsg', 98);

proc ark_corr_uint8_3_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<uint8,3,uint64,2>', ark_corr_uint8_3_uint_2, 'StatsMsg', 98);

proc ark_corr_uint8_3_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<uint8,3,uint64,3>', ark_corr_uint8_3_uint_3, 'StatsMsg', 98);

proc ark_corr_uint8_3_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<uint8,3,uint64,4>', ark_corr_uint8_3_uint_4, 'StatsMsg', 98);

proc ark_corr_uint8_3_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<uint8,3,uint8,1>', ark_corr_uint8_3_uint8_1, 'StatsMsg', 98);

proc ark_corr_uint8_3_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<uint8,3,uint8,2>', ark_corr_uint8_3_uint8_2, 'StatsMsg', 98);

proc ark_corr_uint8_3_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<uint8,3,uint8,3>', ark_corr_uint8_3_uint8_3, 'StatsMsg', 98);

proc ark_corr_uint8_3_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<uint8,3,uint8,4>', ark_corr_uint8_3_uint8_4, 'StatsMsg', 98);

proc ark_corr_uint8_3_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<uint8,3,float64,1>', ark_corr_uint8_3_real_1, 'StatsMsg', 98);

proc ark_corr_uint8_3_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<uint8,3,float64,2>', ark_corr_uint8_3_real_2, 'StatsMsg', 98);

proc ark_corr_uint8_3_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<uint8,3,float64,3>', ark_corr_uint8_3_real_3, 'StatsMsg', 98);

proc ark_corr_uint8_3_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<uint8,3,float64,4>', ark_corr_uint8_3_real_4, 'StatsMsg', 98);

proc ark_corr_uint8_3_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<uint8,3,bool,1>', ark_corr_uint8_3_bool_1, 'StatsMsg', 98);

proc ark_corr_uint8_3_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<uint8,3,bool,2>', ark_corr_uint8_3_bool_2, 'StatsMsg', 98);

proc ark_corr_uint8_3_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<uint8,3,bool,3>', ark_corr_uint8_3_bool_3, 'StatsMsg', 98);

proc ark_corr_uint8_3_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<uint8,3,bool,4>', ark_corr_uint8_3_bool_4, 'StatsMsg', 98);

proc ark_corr_uint8_3_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<uint8,3,bigint,1>', ark_corr_uint8_3_bigint_1, 'StatsMsg', 98);

proc ark_corr_uint8_3_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<uint8,3,bigint,2>', ark_corr_uint8_3_bigint_2, 'StatsMsg', 98);

proc ark_corr_uint8_3_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<uint8,3,bigint,3>', ark_corr_uint8_3_bigint_3, 'StatsMsg', 98);

proc ark_corr_uint8_3_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<uint8,3,bigint,4>', ark_corr_uint8_3_bigint_4, 'StatsMsg', 98);

proc ark_corr_uint8_4_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<uint8,4,int64,1>', ark_corr_uint8_4_int_1, 'StatsMsg', 98);

proc ark_corr_uint8_4_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<uint8,4,int64,2>', ark_corr_uint8_4_int_2, 'StatsMsg', 98);

proc ark_corr_uint8_4_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<uint8,4,int64,3>', ark_corr_uint8_4_int_3, 'StatsMsg', 98);

proc ark_corr_uint8_4_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<uint8,4,int64,4>', ark_corr_uint8_4_int_4, 'StatsMsg', 98);

proc ark_corr_uint8_4_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<uint8,4,uint64,1>', ark_corr_uint8_4_uint_1, 'StatsMsg', 98);

proc ark_corr_uint8_4_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<uint8,4,uint64,2>', ark_corr_uint8_4_uint_2, 'StatsMsg', 98);

proc ark_corr_uint8_4_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<uint8,4,uint64,3>', ark_corr_uint8_4_uint_3, 'StatsMsg', 98);

proc ark_corr_uint8_4_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<uint8,4,uint64,4>', ark_corr_uint8_4_uint_4, 'StatsMsg', 98);

proc ark_corr_uint8_4_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<uint8,4,uint8,1>', ark_corr_uint8_4_uint8_1, 'StatsMsg', 98);

proc ark_corr_uint8_4_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<uint8,4,uint8,2>', ark_corr_uint8_4_uint8_2, 'StatsMsg', 98);

proc ark_corr_uint8_4_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<uint8,4,uint8,3>', ark_corr_uint8_4_uint8_3, 'StatsMsg', 98);

proc ark_corr_uint8_4_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<uint8,4,uint8,4>', ark_corr_uint8_4_uint8_4, 'StatsMsg', 98);

proc ark_corr_uint8_4_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<uint8,4,float64,1>', ark_corr_uint8_4_real_1, 'StatsMsg', 98);

proc ark_corr_uint8_4_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<uint8,4,float64,2>', ark_corr_uint8_4_real_2, 'StatsMsg', 98);

proc ark_corr_uint8_4_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<uint8,4,float64,3>', ark_corr_uint8_4_real_3, 'StatsMsg', 98);

proc ark_corr_uint8_4_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<uint8,4,float64,4>', ark_corr_uint8_4_real_4, 'StatsMsg', 98);

proc ark_corr_uint8_4_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<uint8,4,bool,1>', ark_corr_uint8_4_bool_1, 'StatsMsg', 98);

proc ark_corr_uint8_4_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<uint8,4,bool,2>', ark_corr_uint8_4_bool_2, 'StatsMsg', 98);

proc ark_corr_uint8_4_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<uint8,4,bool,3>', ark_corr_uint8_4_bool_3, 'StatsMsg', 98);

proc ark_corr_uint8_4_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<uint8,4,bool,4>', ark_corr_uint8_4_bool_4, 'StatsMsg', 98);

proc ark_corr_uint8_4_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<uint8,4,bigint,1>', ark_corr_uint8_4_bigint_1, 'StatsMsg', 98);

proc ark_corr_uint8_4_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<uint8,4,bigint,2>', ark_corr_uint8_4_bigint_2, 'StatsMsg', 98);

proc ark_corr_uint8_4_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<uint8,4,bigint,3>', ark_corr_uint8_4_bigint_3, 'StatsMsg', 98);

proc ark_corr_uint8_4_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<uint8,4,bigint,4>', ark_corr_uint8_4_bigint_4, 'StatsMsg', 98);

proc ark_corr_real_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<float64,1,int64,1>', ark_corr_real_1_int_1, 'StatsMsg', 98);

proc ark_corr_real_1_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<float64,1,int64,2>', ark_corr_real_1_int_2, 'StatsMsg', 98);

proc ark_corr_real_1_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<float64,1,int64,3>', ark_corr_real_1_int_3, 'StatsMsg', 98);

proc ark_corr_real_1_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<float64,1,int64,4>', ark_corr_real_1_int_4, 'StatsMsg', 98);

proc ark_corr_real_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<float64,1,uint64,1>', ark_corr_real_1_uint_1, 'StatsMsg', 98);

proc ark_corr_real_1_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<float64,1,uint64,2>', ark_corr_real_1_uint_2, 'StatsMsg', 98);

proc ark_corr_real_1_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<float64,1,uint64,3>', ark_corr_real_1_uint_3, 'StatsMsg', 98);

proc ark_corr_real_1_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<float64,1,uint64,4>', ark_corr_real_1_uint_4, 'StatsMsg', 98);

proc ark_corr_real_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<float64,1,uint8,1>', ark_corr_real_1_uint8_1, 'StatsMsg', 98);

proc ark_corr_real_1_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<float64,1,uint8,2>', ark_corr_real_1_uint8_2, 'StatsMsg', 98);

proc ark_corr_real_1_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<float64,1,uint8,3>', ark_corr_real_1_uint8_3, 'StatsMsg', 98);

proc ark_corr_real_1_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<float64,1,uint8,4>', ark_corr_real_1_uint8_4, 'StatsMsg', 98);

proc ark_corr_real_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<float64,1,float64,1>', ark_corr_real_1_real_1, 'StatsMsg', 98);

proc ark_corr_real_1_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<float64,1,float64,2>', ark_corr_real_1_real_2, 'StatsMsg', 98);

proc ark_corr_real_1_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<float64,1,float64,3>', ark_corr_real_1_real_3, 'StatsMsg', 98);

proc ark_corr_real_1_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<float64,1,float64,4>', ark_corr_real_1_real_4, 'StatsMsg', 98);

proc ark_corr_real_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<float64,1,bool,1>', ark_corr_real_1_bool_1, 'StatsMsg', 98);

proc ark_corr_real_1_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<float64,1,bool,2>', ark_corr_real_1_bool_2, 'StatsMsg', 98);

proc ark_corr_real_1_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<float64,1,bool,3>', ark_corr_real_1_bool_3, 'StatsMsg', 98);

proc ark_corr_real_1_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<float64,1,bool,4>', ark_corr_real_1_bool_4, 'StatsMsg', 98);

proc ark_corr_real_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<float64,1,bigint,1>', ark_corr_real_1_bigint_1, 'StatsMsg', 98);

proc ark_corr_real_1_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<float64,1,bigint,2>', ark_corr_real_1_bigint_2, 'StatsMsg', 98);

proc ark_corr_real_1_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<float64,1,bigint,3>', ark_corr_real_1_bigint_3, 'StatsMsg', 98);

proc ark_corr_real_1_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<float64,1,bigint,4>', ark_corr_real_1_bigint_4, 'StatsMsg', 98);

proc ark_corr_real_2_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<float64,2,int64,1>', ark_corr_real_2_int_1, 'StatsMsg', 98);

proc ark_corr_real_2_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<float64,2,int64,2>', ark_corr_real_2_int_2, 'StatsMsg', 98);

proc ark_corr_real_2_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<float64,2,int64,3>', ark_corr_real_2_int_3, 'StatsMsg', 98);

proc ark_corr_real_2_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<float64,2,int64,4>', ark_corr_real_2_int_4, 'StatsMsg', 98);

proc ark_corr_real_2_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<float64,2,uint64,1>', ark_corr_real_2_uint_1, 'StatsMsg', 98);

proc ark_corr_real_2_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<float64,2,uint64,2>', ark_corr_real_2_uint_2, 'StatsMsg', 98);

proc ark_corr_real_2_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<float64,2,uint64,3>', ark_corr_real_2_uint_3, 'StatsMsg', 98);

proc ark_corr_real_2_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<float64,2,uint64,4>', ark_corr_real_2_uint_4, 'StatsMsg', 98);

proc ark_corr_real_2_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<float64,2,uint8,1>', ark_corr_real_2_uint8_1, 'StatsMsg', 98);

proc ark_corr_real_2_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<float64,2,uint8,2>', ark_corr_real_2_uint8_2, 'StatsMsg', 98);

proc ark_corr_real_2_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<float64,2,uint8,3>', ark_corr_real_2_uint8_3, 'StatsMsg', 98);

proc ark_corr_real_2_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<float64,2,uint8,4>', ark_corr_real_2_uint8_4, 'StatsMsg', 98);

proc ark_corr_real_2_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<float64,2,float64,1>', ark_corr_real_2_real_1, 'StatsMsg', 98);

proc ark_corr_real_2_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<float64,2,float64,2>', ark_corr_real_2_real_2, 'StatsMsg', 98);

proc ark_corr_real_2_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<float64,2,float64,3>', ark_corr_real_2_real_3, 'StatsMsg', 98);

proc ark_corr_real_2_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<float64,2,float64,4>', ark_corr_real_2_real_4, 'StatsMsg', 98);

proc ark_corr_real_2_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<float64,2,bool,1>', ark_corr_real_2_bool_1, 'StatsMsg', 98);

proc ark_corr_real_2_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<float64,2,bool,2>', ark_corr_real_2_bool_2, 'StatsMsg', 98);

proc ark_corr_real_2_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<float64,2,bool,3>', ark_corr_real_2_bool_3, 'StatsMsg', 98);

proc ark_corr_real_2_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<float64,2,bool,4>', ark_corr_real_2_bool_4, 'StatsMsg', 98);

proc ark_corr_real_2_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<float64,2,bigint,1>', ark_corr_real_2_bigint_1, 'StatsMsg', 98);

proc ark_corr_real_2_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<float64,2,bigint,2>', ark_corr_real_2_bigint_2, 'StatsMsg', 98);

proc ark_corr_real_2_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<float64,2,bigint,3>', ark_corr_real_2_bigint_3, 'StatsMsg', 98);

proc ark_corr_real_2_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<float64,2,bigint,4>', ark_corr_real_2_bigint_4, 'StatsMsg', 98);

proc ark_corr_real_3_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<float64,3,int64,1>', ark_corr_real_3_int_1, 'StatsMsg', 98);

proc ark_corr_real_3_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<float64,3,int64,2>', ark_corr_real_3_int_2, 'StatsMsg', 98);

proc ark_corr_real_3_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<float64,3,int64,3>', ark_corr_real_3_int_3, 'StatsMsg', 98);

proc ark_corr_real_3_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<float64,3,int64,4>', ark_corr_real_3_int_4, 'StatsMsg', 98);

proc ark_corr_real_3_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<float64,3,uint64,1>', ark_corr_real_3_uint_1, 'StatsMsg', 98);

proc ark_corr_real_3_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<float64,3,uint64,2>', ark_corr_real_3_uint_2, 'StatsMsg', 98);

proc ark_corr_real_3_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<float64,3,uint64,3>', ark_corr_real_3_uint_3, 'StatsMsg', 98);

proc ark_corr_real_3_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<float64,3,uint64,4>', ark_corr_real_3_uint_4, 'StatsMsg', 98);

proc ark_corr_real_3_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<float64,3,uint8,1>', ark_corr_real_3_uint8_1, 'StatsMsg', 98);

proc ark_corr_real_3_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<float64,3,uint8,2>', ark_corr_real_3_uint8_2, 'StatsMsg', 98);

proc ark_corr_real_3_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<float64,3,uint8,3>', ark_corr_real_3_uint8_3, 'StatsMsg', 98);

proc ark_corr_real_3_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<float64,3,uint8,4>', ark_corr_real_3_uint8_4, 'StatsMsg', 98);

proc ark_corr_real_3_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<float64,3,float64,1>', ark_corr_real_3_real_1, 'StatsMsg', 98);

proc ark_corr_real_3_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<float64,3,float64,2>', ark_corr_real_3_real_2, 'StatsMsg', 98);

proc ark_corr_real_3_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<float64,3,float64,3>', ark_corr_real_3_real_3, 'StatsMsg', 98);

proc ark_corr_real_3_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<float64,3,float64,4>', ark_corr_real_3_real_4, 'StatsMsg', 98);

proc ark_corr_real_3_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<float64,3,bool,1>', ark_corr_real_3_bool_1, 'StatsMsg', 98);

proc ark_corr_real_3_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<float64,3,bool,2>', ark_corr_real_3_bool_2, 'StatsMsg', 98);

proc ark_corr_real_3_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<float64,3,bool,3>', ark_corr_real_3_bool_3, 'StatsMsg', 98);

proc ark_corr_real_3_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<float64,3,bool,4>', ark_corr_real_3_bool_4, 'StatsMsg', 98);

proc ark_corr_real_3_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<float64,3,bigint,1>', ark_corr_real_3_bigint_1, 'StatsMsg', 98);

proc ark_corr_real_3_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<float64,3,bigint,2>', ark_corr_real_3_bigint_2, 'StatsMsg', 98);

proc ark_corr_real_3_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<float64,3,bigint,3>', ark_corr_real_3_bigint_3, 'StatsMsg', 98);

proc ark_corr_real_3_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<float64,3,bigint,4>', ark_corr_real_3_bigint_4, 'StatsMsg', 98);

proc ark_corr_real_4_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<float64,4,int64,1>', ark_corr_real_4_int_1, 'StatsMsg', 98);

proc ark_corr_real_4_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<float64,4,int64,2>', ark_corr_real_4_int_2, 'StatsMsg', 98);

proc ark_corr_real_4_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<float64,4,int64,3>', ark_corr_real_4_int_3, 'StatsMsg', 98);

proc ark_corr_real_4_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<float64,4,int64,4>', ark_corr_real_4_int_4, 'StatsMsg', 98);

proc ark_corr_real_4_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<float64,4,uint64,1>', ark_corr_real_4_uint_1, 'StatsMsg', 98);

proc ark_corr_real_4_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<float64,4,uint64,2>', ark_corr_real_4_uint_2, 'StatsMsg', 98);

proc ark_corr_real_4_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<float64,4,uint64,3>', ark_corr_real_4_uint_3, 'StatsMsg', 98);

proc ark_corr_real_4_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<float64,4,uint64,4>', ark_corr_real_4_uint_4, 'StatsMsg', 98);

proc ark_corr_real_4_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<float64,4,uint8,1>', ark_corr_real_4_uint8_1, 'StatsMsg', 98);

proc ark_corr_real_4_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<float64,4,uint8,2>', ark_corr_real_4_uint8_2, 'StatsMsg', 98);

proc ark_corr_real_4_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<float64,4,uint8,3>', ark_corr_real_4_uint8_3, 'StatsMsg', 98);

proc ark_corr_real_4_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<float64,4,uint8,4>', ark_corr_real_4_uint8_4, 'StatsMsg', 98);

proc ark_corr_real_4_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<float64,4,float64,1>', ark_corr_real_4_real_1, 'StatsMsg', 98);

proc ark_corr_real_4_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<float64,4,float64,2>', ark_corr_real_4_real_2, 'StatsMsg', 98);

proc ark_corr_real_4_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<float64,4,float64,3>', ark_corr_real_4_real_3, 'StatsMsg', 98);

proc ark_corr_real_4_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<float64,4,float64,4>', ark_corr_real_4_real_4, 'StatsMsg', 98);

proc ark_corr_real_4_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<float64,4,bool,1>', ark_corr_real_4_bool_1, 'StatsMsg', 98);

proc ark_corr_real_4_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<float64,4,bool,2>', ark_corr_real_4_bool_2, 'StatsMsg', 98);

proc ark_corr_real_4_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<float64,4,bool,3>', ark_corr_real_4_bool_3, 'StatsMsg', 98);

proc ark_corr_real_4_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<float64,4,bool,4>', ark_corr_real_4_bool_4, 'StatsMsg', 98);

proc ark_corr_real_4_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<float64,4,bigint,1>', ark_corr_real_4_bigint_1, 'StatsMsg', 98);

proc ark_corr_real_4_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<float64,4,bigint,2>', ark_corr_real_4_bigint_2, 'StatsMsg', 98);

proc ark_corr_real_4_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<float64,4,bigint,3>', ark_corr_real_4_bigint_3, 'StatsMsg', 98);

proc ark_corr_real_4_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<float64,4,bigint,4>', ark_corr_real_4_bigint_4, 'StatsMsg', 98);

proc ark_corr_bool_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<bool,1,int64,1>', ark_corr_bool_1_int_1, 'StatsMsg', 98);

proc ark_corr_bool_1_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<bool,1,int64,2>', ark_corr_bool_1_int_2, 'StatsMsg', 98);

proc ark_corr_bool_1_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<bool,1,int64,3>', ark_corr_bool_1_int_3, 'StatsMsg', 98);

proc ark_corr_bool_1_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<bool,1,int64,4>', ark_corr_bool_1_int_4, 'StatsMsg', 98);

proc ark_corr_bool_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<bool,1,uint64,1>', ark_corr_bool_1_uint_1, 'StatsMsg', 98);

proc ark_corr_bool_1_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<bool,1,uint64,2>', ark_corr_bool_1_uint_2, 'StatsMsg', 98);

proc ark_corr_bool_1_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<bool,1,uint64,3>', ark_corr_bool_1_uint_3, 'StatsMsg', 98);

proc ark_corr_bool_1_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<bool,1,uint64,4>', ark_corr_bool_1_uint_4, 'StatsMsg', 98);

proc ark_corr_bool_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<bool,1,uint8,1>', ark_corr_bool_1_uint8_1, 'StatsMsg', 98);

proc ark_corr_bool_1_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<bool,1,uint8,2>', ark_corr_bool_1_uint8_2, 'StatsMsg', 98);

proc ark_corr_bool_1_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<bool,1,uint8,3>', ark_corr_bool_1_uint8_3, 'StatsMsg', 98);

proc ark_corr_bool_1_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<bool,1,uint8,4>', ark_corr_bool_1_uint8_4, 'StatsMsg', 98);

proc ark_corr_bool_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<bool,1,float64,1>', ark_corr_bool_1_real_1, 'StatsMsg', 98);

proc ark_corr_bool_1_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<bool,1,float64,2>', ark_corr_bool_1_real_2, 'StatsMsg', 98);

proc ark_corr_bool_1_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<bool,1,float64,3>', ark_corr_bool_1_real_3, 'StatsMsg', 98);

proc ark_corr_bool_1_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<bool,1,float64,4>', ark_corr_bool_1_real_4, 'StatsMsg', 98);

proc ark_corr_bool_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<bool,1,bool,1>', ark_corr_bool_1_bool_1, 'StatsMsg', 98);

proc ark_corr_bool_1_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<bool,1,bool,2>', ark_corr_bool_1_bool_2, 'StatsMsg', 98);

proc ark_corr_bool_1_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<bool,1,bool,3>', ark_corr_bool_1_bool_3, 'StatsMsg', 98);

proc ark_corr_bool_1_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<bool,1,bool,4>', ark_corr_bool_1_bool_4, 'StatsMsg', 98);

proc ark_corr_bool_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<bool,1,bigint,1>', ark_corr_bool_1_bigint_1, 'StatsMsg', 98);

proc ark_corr_bool_1_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<bool,1,bigint,2>', ark_corr_bool_1_bigint_2, 'StatsMsg', 98);

proc ark_corr_bool_1_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<bool,1,bigint,3>', ark_corr_bool_1_bigint_3, 'StatsMsg', 98);

proc ark_corr_bool_1_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<bool,1,bigint,4>', ark_corr_bool_1_bigint_4, 'StatsMsg', 98);

proc ark_corr_bool_2_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<bool,2,int64,1>', ark_corr_bool_2_int_1, 'StatsMsg', 98);

proc ark_corr_bool_2_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<bool,2,int64,2>', ark_corr_bool_2_int_2, 'StatsMsg', 98);

proc ark_corr_bool_2_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<bool,2,int64,3>', ark_corr_bool_2_int_3, 'StatsMsg', 98);

proc ark_corr_bool_2_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<bool,2,int64,4>', ark_corr_bool_2_int_4, 'StatsMsg', 98);

proc ark_corr_bool_2_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<bool,2,uint64,1>', ark_corr_bool_2_uint_1, 'StatsMsg', 98);

proc ark_corr_bool_2_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<bool,2,uint64,2>', ark_corr_bool_2_uint_2, 'StatsMsg', 98);

proc ark_corr_bool_2_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<bool,2,uint64,3>', ark_corr_bool_2_uint_3, 'StatsMsg', 98);

proc ark_corr_bool_2_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<bool,2,uint64,4>', ark_corr_bool_2_uint_4, 'StatsMsg', 98);

proc ark_corr_bool_2_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<bool,2,uint8,1>', ark_corr_bool_2_uint8_1, 'StatsMsg', 98);

proc ark_corr_bool_2_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<bool,2,uint8,2>', ark_corr_bool_2_uint8_2, 'StatsMsg', 98);

proc ark_corr_bool_2_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<bool,2,uint8,3>', ark_corr_bool_2_uint8_3, 'StatsMsg', 98);

proc ark_corr_bool_2_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<bool,2,uint8,4>', ark_corr_bool_2_uint8_4, 'StatsMsg', 98);

proc ark_corr_bool_2_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<bool,2,float64,1>', ark_corr_bool_2_real_1, 'StatsMsg', 98);

proc ark_corr_bool_2_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<bool,2,float64,2>', ark_corr_bool_2_real_2, 'StatsMsg', 98);

proc ark_corr_bool_2_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<bool,2,float64,3>', ark_corr_bool_2_real_3, 'StatsMsg', 98);

proc ark_corr_bool_2_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<bool,2,float64,4>', ark_corr_bool_2_real_4, 'StatsMsg', 98);

proc ark_corr_bool_2_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<bool,2,bool,1>', ark_corr_bool_2_bool_1, 'StatsMsg', 98);

proc ark_corr_bool_2_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<bool,2,bool,2>', ark_corr_bool_2_bool_2, 'StatsMsg', 98);

proc ark_corr_bool_2_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<bool,2,bool,3>', ark_corr_bool_2_bool_3, 'StatsMsg', 98);

proc ark_corr_bool_2_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<bool,2,bool,4>', ark_corr_bool_2_bool_4, 'StatsMsg', 98);

proc ark_corr_bool_2_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<bool,2,bigint,1>', ark_corr_bool_2_bigint_1, 'StatsMsg', 98);

proc ark_corr_bool_2_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<bool,2,bigint,2>', ark_corr_bool_2_bigint_2, 'StatsMsg', 98);

proc ark_corr_bool_2_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<bool,2,bigint,3>', ark_corr_bool_2_bigint_3, 'StatsMsg', 98);

proc ark_corr_bool_2_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<bool,2,bigint,4>', ark_corr_bool_2_bigint_4, 'StatsMsg', 98);

proc ark_corr_bool_3_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<bool,3,int64,1>', ark_corr_bool_3_int_1, 'StatsMsg', 98);

proc ark_corr_bool_3_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<bool,3,int64,2>', ark_corr_bool_3_int_2, 'StatsMsg', 98);

proc ark_corr_bool_3_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<bool,3,int64,3>', ark_corr_bool_3_int_3, 'StatsMsg', 98);

proc ark_corr_bool_3_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<bool,3,int64,4>', ark_corr_bool_3_int_4, 'StatsMsg', 98);

proc ark_corr_bool_3_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<bool,3,uint64,1>', ark_corr_bool_3_uint_1, 'StatsMsg', 98);

proc ark_corr_bool_3_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<bool,3,uint64,2>', ark_corr_bool_3_uint_2, 'StatsMsg', 98);

proc ark_corr_bool_3_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<bool,3,uint64,3>', ark_corr_bool_3_uint_3, 'StatsMsg', 98);

proc ark_corr_bool_3_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<bool,3,uint64,4>', ark_corr_bool_3_uint_4, 'StatsMsg', 98);

proc ark_corr_bool_3_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<bool,3,uint8,1>', ark_corr_bool_3_uint8_1, 'StatsMsg', 98);

proc ark_corr_bool_3_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<bool,3,uint8,2>', ark_corr_bool_3_uint8_2, 'StatsMsg', 98);

proc ark_corr_bool_3_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<bool,3,uint8,3>', ark_corr_bool_3_uint8_3, 'StatsMsg', 98);

proc ark_corr_bool_3_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<bool,3,uint8,4>', ark_corr_bool_3_uint8_4, 'StatsMsg', 98);

proc ark_corr_bool_3_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<bool,3,float64,1>', ark_corr_bool_3_real_1, 'StatsMsg', 98);

proc ark_corr_bool_3_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<bool,3,float64,2>', ark_corr_bool_3_real_2, 'StatsMsg', 98);

proc ark_corr_bool_3_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<bool,3,float64,3>', ark_corr_bool_3_real_3, 'StatsMsg', 98);

proc ark_corr_bool_3_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<bool,3,float64,4>', ark_corr_bool_3_real_4, 'StatsMsg', 98);

proc ark_corr_bool_3_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<bool,3,bool,1>', ark_corr_bool_3_bool_1, 'StatsMsg', 98);

proc ark_corr_bool_3_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<bool,3,bool,2>', ark_corr_bool_3_bool_2, 'StatsMsg', 98);

proc ark_corr_bool_3_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<bool,3,bool,3>', ark_corr_bool_3_bool_3, 'StatsMsg', 98);

proc ark_corr_bool_3_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<bool,3,bool,4>', ark_corr_bool_3_bool_4, 'StatsMsg', 98);

proc ark_corr_bool_3_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<bool,3,bigint,1>', ark_corr_bool_3_bigint_1, 'StatsMsg', 98);

proc ark_corr_bool_3_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<bool,3,bigint,2>', ark_corr_bool_3_bigint_2, 'StatsMsg', 98);

proc ark_corr_bool_3_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<bool,3,bigint,3>', ark_corr_bool_3_bigint_3, 'StatsMsg', 98);

proc ark_corr_bool_3_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<bool,3,bigint,4>', ark_corr_bool_3_bigint_4, 'StatsMsg', 98);

proc ark_corr_bool_4_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<bool,4,int64,1>', ark_corr_bool_4_int_1, 'StatsMsg', 98);

proc ark_corr_bool_4_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<bool,4,int64,2>', ark_corr_bool_4_int_2, 'StatsMsg', 98);

proc ark_corr_bool_4_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<bool,4,int64,3>', ark_corr_bool_4_int_3, 'StatsMsg', 98);

proc ark_corr_bool_4_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<bool,4,int64,4>', ark_corr_bool_4_int_4, 'StatsMsg', 98);

proc ark_corr_bool_4_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<bool,4,uint64,1>', ark_corr_bool_4_uint_1, 'StatsMsg', 98);

proc ark_corr_bool_4_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<bool,4,uint64,2>', ark_corr_bool_4_uint_2, 'StatsMsg', 98);

proc ark_corr_bool_4_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<bool,4,uint64,3>', ark_corr_bool_4_uint_3, 'StatsMsg', 98);

proc ark_corr_bool_4_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<bool,4,uint64,4>', ark_corr_bool_4_uint_4, 'StatsMsg', 98);

proc ark_corr_bool_4_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<bool,4,uint8,1>', ark_corr_bool_4_uint8_1, 'StatsMsg', 98);

proc ark_corr_bool_4_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<bool,4,uint8,2>', ark_corr_bool_4_uint8_2, 'StatsMsg', 98);

proc ark_corr_bool_4_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<bool,4,uint8,3>', ark_corr_bool_4_uint8_3, 'StatsMsg', 98);

proc ark_corr_bool_4_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<bool,4,uint8,4>', ark_corr_bool_4_uint8_4, 'StatsMsg', 98);

proc ark_corr_bool_4_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<bool,4,float64,1>', ark_corr_bool_4_real_1, 'StatsMsg', 98);

proc ark_corr_bool_4_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<bool,4,float64,2>', ark_corr_bool_4_real_2, 'StatsMsg', 98);

proc ark_corr_bool_4_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<bool,4,float64,3>', ark_corr_bool_4_real_3, 'StatsMsg', 98);

proc ark_corr_bool_4_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<bool,4,float64,4>', ark_corr_bool_4_real_4, 'StatsMsg', 98);

proc ark_corr_bool_4_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<bool,4,bool,1>', ark_corr_bool_4_bool_1, 'StatsMsg', 98);

proc ark_corr_bool_4_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<bool,4,bool,2>', ark_corr_bool_4_bool_2, 'StatsMsg', 98);

proc ark_corr_bool_4_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<bool,4,bool,3>', ark_corr_bool_4_bool_3, 'StatsMsg', 98);

proc ark_corr_bool_4_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<bool,4,bool,4>', ark_corr_bool_4_bool_4, 'StatsMsg', 98);

proc ark_corr_bool_4_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<bool,4,bigint,1>', ark_corr_bool_4_bigint_1, 'StatsMsg', 98);

proc ark_corr_bool_4_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<bool,4,bigint,2>', ark_corr_bool_4_bigint_2, 'StatsMsg', 98);

proc ark_corr_bool_4_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<bool,4,bigint,3>', ark_corr_bool_4_bigint_3, 'StatsMsg', 98);

proc ark_corr_bool_4_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<bool,4,bigint,4>', ark_corr_bool_4_bigint_4, 'StatsMsg', 98);

proc ark_corr_bigint_1_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<bigint,1,int64,1>', ark_corr_bigint_1_int_1, 'StatsMsg', 98);

proc ark_corr_bigint_1_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<bigint,1,int64,2>', ark_corr_bigint_1_int_2, 'StatsMsg', 98);

proc ark_corr_bigint_1_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<bigint,1,int64,3>', ark_corr_bigint_1_int_3, 'StatsMsg', 98);

proc ark_corr_bigint_1_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<bigint,1,int64,4>', ark_corr_bigint_1_int_4, 'StatsMsg', 98);

proc ark_corr_bigint_1_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<bigint,1,uint64,1>', ark_corr_bigint_1_uint_1, 'StatsMsg', 98);

proc ark_corr_bigint_1_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<bigint,1,uint64,2>', ark_corr_bigint_1_uint_2, 'StatsMsg', 98);

proc ark_corr_bigint_1_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<bigint,1,uint64,3>', ark_corr_bigint_1_uint_3, 'StatsMsg', 98);

proc ark_corr_bigint_1_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<bigint,1,uint64,4>', ark_corr_bigint_1_uint_4, 'StatsMsg', 98);

proc ark_corr_bigint_1_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<bigint,1,uint8,1>', ark_corr_bigint_1_uint8_1, 'StatsMsg', 98);

proc ark_corr_bigint_1_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<bigint,1,uint8,2>', ark_corr_bigint_1_uint8_2, 'StatsMsg', 98);

proc ark_corr_bigint_1_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<bigint,1,uint8,3>', ark_corr_bigint_1_uint8_3, 'StatsMsg', 98);

proc ark_corr_bigint_1_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<bigint,1,uint8,4>', ark_corr_bigint_1_uint8_4, 'StatsMsg', 98);

proc ark_corr_bigint_1_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<bigint,1,float64,1>', ark_corr_bigint_1_real_1, 'StatsMsg', 98);

proc ark_corr_bigint_1_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<bigint,1,float64,2>', ark_corr_bigint_1_real_2, 'StatsMsg', 98);

proc ark_corr_bigint_1_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<bigint,1,float64,3>', ark_corr_bigint_1_real_3, 'StatsMsg', 98);

proc ark_corr_bigint_1_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<bigint,1,float64,4>', ark_corr_bigint_1_real_4, 'StatsMsg', 98);

proc ark_corr_bigint_1_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<bigint,1,bool,1>', ark_corr_bigint_1_bool_1, 'StatsMsg', 98);

proc ark_corr_bigint_1_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<bigint,1,bool,2>', ark_corr_bigint_1_bool_2, 'StatsMsg', 98);

proc ark_corr_bigint_1_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<bigint,1,bool,3>', ark_corr_bigint_1_bool_3, 'StatsMsg', 98);

proc ark_corr_bigint_1_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<bigint,1,bool,4>', ark_corr_bigint_1_bool_4, 'StatsMsg', 98);

proc ark_corr_bigint_1_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<bigint,1,bigint,1>', ark_corr_bigint_1_bigint_1, 'StatsMsg', 98);

proc ark_corr_bigint_1_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<bigint,1,bigint,2>', ark_corr_bigint_1_bigint_2, 'StatsMsg', 98);

proc ark_corr_bigint_1_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<bigint,1,bigint,3>', ark_corr_bigint_1_bigint_3, 'StatsMsg', 98);

proc ark_corr_bigint_1_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<bigint,1,bigint,4>', ark_corr_bigint_1_bigint_4, 'StatsMsg', 98);

proc ark_corr_bigint_2_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<bigint,2,int64,1>', ark_corr_bigint_2_int_1, 'StatsMsg', 98);

proc ark_corr_bigint_2_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<bigint,2,int64,2>', ark_corr_bigint_2_int_2, 'StatsMsg', 98);

proc ark_corr_bigint_2_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<bigint,2,int64,3>', ark_corr_bigint_2_int_3, 'StatsMsg', 98);

proc ark_corr_bigint_2_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<bigint,2,int64,4>', ark_corr_bigint_2_int_4, 'StatsMsg', 98);

proc ark_corr_bigint_2_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<bigint,2,uint64,1>', ark_corr_bigint_2_uint_1, 'StatsMsg', 98);

proc ark_corr_bigint_2_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<bigint,2,uint64,2>', ark_corr_bigint_2_uint_2, 'StatsMsg', 98);

proc ark_corr_bigint_2_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<bigint,2,uint64,3>', ark_corr_bigint_2_uint_3, 'StatsMsg', 98);

proc ark_corr_bigint_2_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<bigint,2,uint64,4>', ark_corr_bigint_2_uint_4, 'StatsMsg', 98);

proc ark_corr_bigint_2_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<bigint,2,uint8,1>', ark_corr_bigint_2_uint8_1, 'StatsMsg', 98);

proc ark_corr_bigint_2_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<bigint,2,uint8,2>', ark_corr_bigint_2_uint8_2, 'StatsMsg', 98);

proc ark_corr_bigint_2_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<bigint,2,uint8,3>', ark_corr_bigint_2_uint8_3, 'StatsMsg', 98);

proc ark_corr_bigint_2_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<bigint,2,uint8,4>', ark_corr_bigint_2_uint8_4, 'StatsMsg', 98);

proc ark_corr_bigint_2_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<bigint,2,float64,1>', ark_corr_bigint_2_real_1, 'StatsMsg', 98);

proc ark_corr_bigint_2_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<bigint,2,float64,2>', ark_corr_bigint_2_real_2, 'StatsMsg', 98);

proc ark_corr_bigint_2_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<bigint,2,float64,3>', ark_corr_bigint_2_real_3, 'StatsMsg', 98);

proc ark_corr_bigint_2_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<bigint,2,float64,4>', ark_corr_bigint_2_real_4, 'StatsMsg', 98);

proc ark_corr_bigint_2_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<bigint,2,bool,1>', ark_corr_bigint_2_bool_1, 'StatsMsg', 98);

proc ark_corr_bigint_2_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<bigint,2,bool,2>', ark_corr_bigint_2_bool_2, 'StatsMsg', 98);

proc ark_corr_bigint_2_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<bigint,2,bool,3>', ark_corr_bigint_2_bool_3, 'StatsMsg', 98);

proc ark_corr_bigint_2_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<bigint,2,bool,4>', ark_corr_bigint_2_bool_4, 'StatsMsg', 98);

proc ark_corr_bigint_2_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<bigint,2,bigint,1>', ark_corr_bigint_2_bigint_1, 'StatsMsg', 98);

proc ark_corr_bigint_2_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<bigint,2,bigint,2>', ark_corr_bigint_2_bigint_2, 'StatsMsg', 98);

proc ark_corr_bigint_2_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<bigint,2,bigint,3>', ark_corr_bigint_2_bigint_3, 'StatsMsg', 98);

proc ark_corr_bigint_2_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<bigint,2,bigint,4>', ark_corr_bigint_2_bigint_4, 'StatsMsg', 98);

proc ark_corr_bigint_3_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<bigint,3,int64,1>', ark_corr_bigint_3_int_1, 'StatsMsg', 98);

proc ark_corr_bigint_3_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<bigint,3,int64,2>', ark_corr_bigint_3_int_2, 'StatsMsg', 98);

proc ark_corr_bigint_3_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<bigint,3,int64,3>', ark_corr_bigint_3_int_3, 'StatsMsg', 98);

proc ark_corr_bigint_3_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<bigint,3,int64,4>', ark_corr_bigint_3_int_4, 'StatsMsg', 98);

proc ark_corr_bigint_3_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<bigint,3,uint64,1>', ark_corr_bigint_3_uint_1, 'StatsMsg', 98);

proc ark_corr_bigint_3_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<bigint,3,uint64,2>', ark_corr_bigint_3_uint_2, 'StatsMsg', 98);

proc ark_corr_bigint_3_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<bigint,3,uint64,3>', ark_corr_bigint_3_uint_3, 'StatsMsg', 98);

proc ark_corr_bigint_3_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<bigint,3,uint64,4>', ark_corr_bigint_3_uint_4, 'StatsMsg', 98);

proc ark_corr_bigint_3_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<bigint,3,uint8,1>', ark_corr_bigint_3_uint8_1, 'StatsMsg', 98);

proc ark_corr_bigint_3_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<bigint,3,uint8,2>', ark_corr_bigint_3_uint8_2, 'StatsMsg', 98);

proc ark_corr_bigint_3_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<bigint,3,uint8,3>', ark_corr_bigint_3_uint8_3, 'StatsMsg', 98);

proc ark_corr_bigint_3_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<bigint,3,uint8,4>', ark_corr_bigint_3_uint8_4, 'StatsMsg', 98);

proc ark_corr_bigint_3_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<bigint,3,float64,1>', ark_corr_bigint_3_real_1, 'StatsMsg', 98);

proc ark_corr_bigint_3_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<bigint,3,float64,2>', ark_corr_bigint_3_real_2, 'StatsMsg', 98);

proc ark_corr_bigint_3_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<bigint,3,float64,3>', ark_corr_bigint_3_real_3, 'StatsMsg', 98);

proc ark_corr_bigint_3_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<bigint,3,float64,4>', ark_corr_bigint_3_real_4, 'StatsMsg', 98);

proc ark_corr_bigint_3_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<bigint,3,bool,1>', ark_corr_bigint_3_bool_1, 'StatsMsg', 98);

proc ark_corr_bigint_3_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<bigint,3,bool,2>', ark_corr_bigint_3_bool_2, 'StatsMsg', 98);

proc ark_corr_bigint_3_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<bigint,3,bool,3>', ark_corr_bigint_3_bool_3, 'StatsMsg', 98);

proc ark_corr_bigint_3_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<bigint,3,bool,4>', ark_corr_bigint_3_bool_4, 'StatsMsg', 98);

proc ark_corr_bigint_3_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<bigint,3,bigint,1>', ark_corr_bigint_3_bigint_1, 'StatsMsg', 98);

proc ark_corr_bigint_3_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<bigint,3,bigint,2>', ark_corr_bigint_3_bigint_2, 'StatsMsg', 98);

proc ark_corr_bigint_3_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<bigint,3,bigint,3>', ark_corr_bigint_3_bigint_3, 'StatsMsg', 98);

proc ark_corr_bigint_3_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<bigint,3,bigint,4>', ark_corr_bigint_3_bigint_4, 'StatsMsg', 98);

proc ark_corr_bigint_4_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=int, array_nd_1=1);
registerFunction('corr<bigint,4,int64,1>', ark_corr_bigint_4_int_1, 'StatsMsg', 98);

proc ark_corr_bigint_4_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=int, array_nd_1=2);
registerFunction('corr<bigint,4,int64,2>', ark_corr_bigint_4_int_2, 'StatsMsg', 98);

proc ark_corr_bigint_4_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=int, array_nd_1=3);
registerFunction('corr<bigint,4,int64,3>', ark_corr_bigint_4_int_3, 'StatsMsg', 98);

proc ark_corr_bigint_4_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=int, array_nd_1=4);
registerFunction('corr<bigint,4,int64,4>', ark_corr_bigint_4_int_4, 'StatsMsg', 98);

proc ark_corr_bigint_4_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint, array_nd_1=1);
registerFunction('corr<bigint,4,uint64,1>', ark_corr_bigint_4_uint_1, 'StatsMsg', 98);

proc ark_corr_bigint_4_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint, array_nd_1=2);
registerFunction('corr<bigint,4,uint64,2>', ark_corr_bigint_4_uint_2, 'StatsMsg', 98);

proc ark_corr_bigint_4_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint, array_nd_1=3);
registerFunction('corr<bigint,4,uint64,3>', ark_corr_bigint_4_uint_3, 'StatsMsg', 98);

proc ark_corr_bigint_4_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint, array_nd_1=4);
registerFunction('corr<bigint,4,uint64,4>', ark_corr_bigint_4_uint_4, 'StatsMsg', 98);

proc ark_corr_bigint_4_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=1);
registerFunction('corr<bigint,4,uint8,1>', ark_corr_bigint_4_uint8_1, 'StatsMsg', 98);

proc ark_corr_bigint_4_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=2);
registerFunction('corr<bigint,4,uint8,2>', ark_corr_bigint_4_uint8_2, 'StatsMsg', 98);

proc ark_corr_bigint_4_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=3);
registerFunction('corr<bigint,4,uint8,3>', ark_corr_bigint_4_uint8_3, 'StatsMsg', 98);

proc ark_corr_bigint_4_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=uint(8), array_nd_1=4);
registerFunction('corr<bigint,4,uint8,4>', ark_corr_bigint_4_uint8_4, 'StatsMsg', 98);

proc ark_corr_bigint_4_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=real, array_nd_1=1);
registerFunction('corr<bigint,4,float64,1>', ark_corr_bigint_4_real_1, 'StatsMsg', 98);

proc ark_corr_bigint_4_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=real, array_nd_1=2);
registerFunction('corr<bigint,4,float64,2>', ark_corr_bigint_4_real_2, 'StatsMsg', 98);

proc ark_corr_bigint_4_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=real, array_nd_1=3);
registerFunction('corr<bigint,4,float64,3>', ark_corr_bigint_4_real_3, 'StatsMsg', 98);

proc ark_corr_bigint_4_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=real, array_nd_1=4);
registerFunction('corr<bigint,4,float64,4>', ark_corr_bigint_4_real_4, 'StatsMsg', 98);

proc ark_corr_bigint_4_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bool, array_nd_1=1);
registerFunction('corr<bigint,4,bool,1>', ark_corr_bigint_4_bool_1, 'StatsMsg', 98);

proc ark_corr_bigint_4_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bool, array_nd_1=2);
registerFunction('corr<bigint,4,bool,2>', ark_corr_bigint_4_bool_2, 'StatsMsg', 98);

proc ark_corr_bigint_4_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bool, array_nd_1=3);
registerFunction('corr<bigint,4,bool,3>', ark_corr_bigint_4_bool_3, 'StatsMsg', 98);

proc ark_corr_bigint_4_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bool, array_nd_1=4);
registerFunction('corr<bigint,4,bool,4>', ark_corr_bigint_4_bool_4, 'StatsMsg', 98);

proc ark_corr_bigint_4_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=1);
registerFunction('corr<bigint,4,bigint,1>', ark_corr_bigint_4_bigint_1, 'StatsMsg', 98);

proc ark_corr_bigint_4_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=2);
registerFunction('corr<bigint,4,bigint,2>', ark_corr_bigint_4_bigint_2, 'StatsMsg', 98);

proc ark_corr_bigint_4_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=3);
registerFunction('corr<bigint,4,bigint,3>', ark_corr_bigint_4_bigint_3, 'StatsMsg', 98);

proc ark_corr_bigint_4_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_corr_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4, array_dtype_1=bigint, array_nd_1=4);
registerFunction('corr<bigint,4,bigint,4>', ark_corr_bigint_4_bigint_4, 'StatsMsg', 98);

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

proc ark_cumSum_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=2);
registerFunction('cumSum<int64,2>', ark_cumSum_int_2, 'StatsMsg', 120);

proc ark_cumSum_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=3);
registerFunction('cumSum<int64,3>', ark_cumSum_int_3, 'StatsMsg', 120);

proc ark_cumSum_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=4);
registerFunction('cumSum<int64,4>', ark_cumSum_int_4, 'StatsMsg', 120);

proc ark_cumSum_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('cumSum<uint64,1>', ark_cumSum_uint_1, 'StatsMsg', 120);

proc ark_cumSum_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=2);
registerFunction('cumSum<uint64,2>', ark_cumSum_uint_2, 'StatsMsg', 120);

proc ark_cumSum_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=3);
registerFunction('cumSum<uint64,3>', ark_cumSum_uint_3, 'StatsMsg', 120);

proc ark_cumSum_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=4);
registerFunction('cumSum<uint64,4>', ark_cumSum_uint_4, 'StatsMsg', 120);

proc ark_cumSum_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('cumSum<uint8,1>', ark_cumSum_uint8_1, 'StatsMsg', 120);

proc ark_cumSum_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=2);
registerFunction('cumSum<uint8,2>', ark_cumSum_uint8_2, 'StatsMsg', 120);

proc ark_cumSum_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=3);
registerFunction('cumSum<uint8,3>', ark_cumSum_uint8_3, 'StatsMsg', 120);

proc ark_cumSum_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=4);
registerFunction('cumSum<uint8,4>', ark_cumSum_uint8_4, 'StatsMsg', 120);

proc ark_cumSum_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('cumSum<float64,1>', ark_cumSum_real_1, 'StatsMsg', 120);

proc ark_cumSum_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=2);
registerFunction('cumSum<float64,2>', ark_cumSum_real_2, 'StatsMsg', 120);

proc ark_cumSum_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=3);
registerFunction('cumSum<float64,3>', ark_cumSum_real_3, 'StatsMsg', 120);

proc ark_cumSum_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=4);
registerFunction('cumSum<float64,4>', ark_cumSum_real_4, 'StatsMsg', 120);

proc ark_cumSum_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('cumSum<bool,1>', ark_cumSum_bool_1, 'StatsMsg', 120);

proc ark_cumSum_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=2);
registerFunction('cumSum<bool,2>', ark_cumSum_bool_2, 'StatsMsg', 120);

proc ark_cumSum_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=3);
registerFunction('cumSum<bool,3>', ark_cumSum_bool_3, 'StatsMsg', 120);

proc ark_cumSum_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=4);
registerFunction('cumSum<bool,4>', ark_cumSum_bool_4, 'StatsMsg', 120);

proc ark_cumSum_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('cumSum<bigint,1>', ark_cumSum_bigint_1, 'StatsMsg', 120);

proc ark_cumSum_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=2);
registerFunction('cumSum<bigint,2>', ark_cumSum_bigint_2, 'StatsMsg', 120);

proc ark_cumSum_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=3);
registerFunction('cumSum<bigint,3>', ark_cumSum_bigint_3, 'StatsMsg', 120);

proc ark_cumSum_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_cumSum_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=4);
registerFunction('cumSum<bigint,4>', ark_cumSum_bigint_4, 'StatsMsg', 120);

import MsgProcessing;

proc ark_create_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('create<int64,1>', ark_create_int_1, 'MsgProcessing', 35);

proc ark_create_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('create<int64,2>', ark_create_int_2, 'MsgProcessing', 35);

proc ark_create_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('create<int64,3>', ark_create_int_3, 'MsgProcessing', 35);

proc ark_create_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('create<int64,4>', ark_create_int_4, 'MsgProcessing', 35);

proc ark_create_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('create<uint64,1>', ark_create_uint_1, 'MsgProcessing', 35);

proc ark_create_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('create<uint64,2>', ark_create_uint_2, 'MsgProcessing', 35);

proc ark_create_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('create<uint64,3>', ark_create_uint_3, 'MsgProcessing', 35);

proc ark_create_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('create<uint64,4>', ark_create_uint_4, 'MsgProcessing', 35);

proc ark_create_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('create<uint8,1>', ark_create_uint8_1, 'MsgProcessing', 35);

proc ark_create_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('create<uint8,2>', ark_create_uint8_2, 'MsgProcessing', 35);

proc ark_create_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('create<uint8,3>', ark_create_uint8_3, 'MsgProcessing', 35);

proc ark_create_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('create<uint8,4>', ark_create_uint8_4, 'MsgProcessing', 35);

proc ark_create_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('create<float64,1>', ark_create_real_1, 'MsgProcessing', 35);

proc ark_create_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('create<float64,2>', ark_create_real_2, 'MsgProcessing', 35);

proc ark_create_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('create<float64,3>', ark_create_real_3, 'MsgProcessing', 35);

proc ark_create_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('create<float64,4>', ark_create_real_4, 'MsgProcessing', 35);

proc ark_create_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('create<bool,1>', ark_create_bool_1, 'MsgProcessing', 35);

proc ark_create_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('create<bool,2>', ark_create_bool_2, 'MsgProcessing', 35);

proc ark_create_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('create<bool,3>', ark_create_bool_3, 'MsgProcessing', 35);

proc ark_create_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('create<bool,4>', ark_create_bool_4, 'MsgProcessing', 35);

proc ark_create_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('create<bigint,1>', ark_create_bigint_1, 'MsgProcessing', 35);

proc ark_create_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('create<bigint,2>', ark_create_bigint_2, 'MsgProcessing', 35);

proc ark_create_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('create<bigint,3>', ark_create_bigint_3, 'MsgProcessing', 35);

proc ark_create_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.create(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('create<bigint,4>', ark_create_bigint_4, 'MsgProcessing', 35);

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

proc ark_set_int_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=int, array_nd=2);
registerFunction('set<int64,2>', ark_set_int_2, 'MsgProcessing', 299);

proc ark_set_int_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=int, array_nd=3);
registerFunction('set<int64,3>', ark_set_int_3, 'MsgProcessing', 299);

proc ark_set_int_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=int, array_nd=4);
registerFunction('set<int64,4>', ark_set_int_4, 'MsgProcessing', 299);

proc ark_set_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('set<uint64,1>', ark_set_uint_1, 'MsgProcessing', 299);

proc ark_set_uint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=2);
registerFunction('set<uint64,2>', ark_set_uint_2, 'MsgProcessing', 299);

proc ark_set_uint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=3);
registerFunction('set<uint64,3>', ark_set_uint_3, 'MsgProcessing', 299);

proc ark_set_uint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=4);
registerFunction('set<uint64,4>', ark_set_uint_4, 'MsgProcessing', 299);

proc ark_set_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=1);
registerFunction('set<uint8,1>', ark_set_uint8_1, 'MsgProcessing', 299);

proc ark_set_uint8_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=2);
registerFunction('set<uint8,2>', ark_set_uint8_2, 'MsgProcessing', 299);

proc ark_set_uint8_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=3);
registerFunction('set<uint8,3>', ark_set_uint8_3, 'MsgProcessing', 299);

proc ark_set_uint8_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=uint(8), array_nd=4);
registerFunction('set<uint8,4>', ark_set_uint8_4, 'MsgProcessing', 299);

proc ark_set_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('set<float64,1>', ark_set_real_1, 'MsgProcessing', 299);

proc ark_set_real_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=real, array_nd=2);
registerFunction('set<float64,2>', ark_set_real_2, 'MsgProcessing', 299);

proc ark_set_real_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=real, array_nd=3);
registerFunction('set<float64,3>', ark_set_real_3, 'MsgProcessing', 299);

proc ark_set_real_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=real, array_nd=4);
registerFunction('set<float64,4>', ark_set_real_4, 'MsgProcessing', 299);

proc ark_set_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('set<bool,1>', ark_set_bool_1, 'MsgProcessing', 299);

proc ark_set_bool_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=2);
registerFunction('set<bool,2>', ark_set_bool_2, 'MsgProcessing', 299);

proc ark_set_bool_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=3);
registerFunction('set<bool,3>', ark_set_bool_3, 'MsgProcessing', 299);

proc ark_set_bool_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=4);
registerFunction('set<bool,4>', ark_set_bool_4, 'MsgProcessing', 299);

proc ark_set_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=1);
registerFunction('set<bigint,1>', ark_set_bigint_1, 'MsgProcessing', 299);

proc ark_set_bigint_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=2);
registerFunction('set<bigint,2>', ark_set_bigint_2, 'MsgProcessing', 299);

proc ark_set_bigint_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=3);
registerFunction('set<bigint,3>', ark_set_bigint_3, 'MsgProcessing', 299);

proc ark_set_bigint_4(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return MsgProcessing.setMsg(cmd, msgArgs, st, array_dtype=bigint, array_nd=4);
registerFunction('set<bigint,4>', ark_set_bigint_4, 'MsgProcessing', 299);

}
