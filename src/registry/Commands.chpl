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

proc ark_reg_sliceIndex_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype_0, param array_nd_0: int): MsgTuple throws {
  var array_array_sym = st[msgArgs['array']]: SymEntry(array_dtype_0, array_nd_0);
  ref array = array_array_sym.a;
  var starts = msgArgs['starts'].toScalarTuple(int, array_nd_0);
  var stops = msgArgs['stops'].toScalarTuple(int, array_nd_0);
  var strides = msgArgs['strides'].toScalarTuple(int, array_nd_0);
  var ark_result = IndexingMsg.sliceIndex(array,starts,stops,strides);
  var ark_result_symbol = new shared SymEntry(ark_result);

  return st.insert(ark_result_symbol);
}

proc ark__slice__int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=int, array_nd_0=1);
registerFunction('[slice]<int64,1>', ark__slice__int_1, 'IndexingMsg', 214);

proc ark__slice__uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('[slice]<uint64,1>', ark__slice__uint_1, 'IndexingMsg', 214);

proc ark__slice__uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('[slice]<uint8,1>', ark__slice__uint8_1, 'IndexingMsg', 214);

proc ark__slice__real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('[slice]<float64,1>', ark__slice__real_1, 'IndexingMsg', 214);

proc ark__slice__bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('[slice]<bool,1>', ark__slice__bool_1, 'IndexingMsg', 214);

proc ark__slice__bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_sliceIndex_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('[slice]<bigint,1>', ark__slice__bigint_1, 'IndexingMsg', 214);

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
registerFunction('[int]=val<int64,1>', ark__int__val_int_1, 'IndexingMsg', 507);

proc ark__int__val_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('[int]=val<uint64,1>', ark__int__val_uint_1, 'IndexingMsg', 507);

proc ark__int__val_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('[int]=val<uint8,1>', ark__int__val_uint8_1, 'IndexingMsg', 507);

proc ark__int__val_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('[int]=val<float64,1>', ark__int__val_real_1, 'IndexingMsg', 507);

proc ark__int__val_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('[int]=val<bool,1>', ark__int__val_bool_1, 'IndexingMsg', 507);

proc ark__int__val_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('[int]=val<bigint,1>', ark__int__val_bigint_1, 'IndexingMsg', 507);

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
registerFunction('[slice]=val<int64,1>', ark__slice__val_int_1, 'IndexingMsg', 919);

proc ark__slice__val_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('[slice]=val<uint64,1>', ark__slice__val_uint_1, 'IndexingMsg', 919);

proc ark__slice__val_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=uint(8), array_nd_0=1);
registerFunction('[slice]=val<uint8,1>', ark__slice__val_uint8_1, 'IndexingMsg', 919);

proc ark__slice__val_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('[slice]=val<float64,1>', ark__slice__val_real_1, 'IndexingMsg', 919);

proc ark__slice__val_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('[slice]=val<bool,1>', ark__slice__val_bool_1, 'IndexingMsg', 919);

proc ark__slice__val_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_setSliceIndexToValue_generic(cmd, msgArgs, st, array_dtype_0=bigint, array_nd_0=1);
registerFunction('[slice]=val<bigint,1>', ark__slice__val_bigint_1, 'IndexingMsg', 919);

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

proc ark__pdarray__int_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=int, array_nd=1);
registerFunction('[pdarray]<int64,int64,1>', ark__pdarray__int_int_1, 'IndexingMsg', 249);

proc ark__pdarray__int_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=uint, array_nd=1);
registerFunction('[pdarray]<int64,uint64,1>', ark__pdarray__int_uint_1, 'IndexingMsg', 249);

proc ark__pdarray__int_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=uint(8), array_nd=1);
registerFunction('[pdarray]<int64,uint8,1>', ark__pdarray__int_uint8_1, 'IndexingMsg', 249);

proc ark__pdarray__int_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=real, array_nd=1);
registerFunction('[pdarray]<int64,float64,1>', ark__pdarray__int_real_1, 'IndexingMsg', 249);

proc ark__pdarray__int_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=bool, array_nd=1);
registerFunction('[pdarray]<int64,bool,1>', ark__pdarray__int_bool_1, 'IndexingMsg', 249);

proc ark__pdarray__int_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=int, array_dtype_idx=bigint, array_nd=1);
registerFunction('[pdarray]<int64,bigint,1>', ark__pdarray__int_bigint_1, 'IndexingMsg', 249);

proc ark__pdarray__uint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=int, array_nd=1);
registerFunction('[pdarray]<uint64,int64,1>', ark__pdarray__uint_int_1, 'IndexingMsg', 249);

proc ark__pdarray__uint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=uint, array_nd=1);
registerFunction('[pdarray]<uint64,uint64,1>', ark__pdarray__uint_uint_1, 'IndexingMsg', 249);

proc ark__pdarray__uint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=uint(8), array_nd=1);
registerFunction('[pdarray]<uint64,uint8,1>', ark__pdarray__uint_uint8_1, 'IndexingMsg', 249);

proc ark__pdarray__uint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=real, array_nd=1);
registerFunction('[pdarray]<uint64,float64,1>', ark__pdarray__uint_real_1, 'IndexingMsg', 249);

proc ark__pdarray__uint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=bool, array_nd=1);
registerFunction('[pdarray]<uint64,bool,1>', ark__pdarray__uint_bool_1, 'IndexingMsg', 249);

proc ark__pdarray__uint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_idx=bigint, array_nd=1);
registerFunction('[pdarray]<uint64,bigint,1>', ark__pdarray__uint_bigint_1, 'IndexingMsg', 249);

proc ark__pdarray__uint8_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=int, array_nd=1);
registerFunction('[pdarray]<uint8,int64,1>', ark__pdarray__uint8_int_1, 'IndexingMsg', 249);

proc ark__pdarray__uint8_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=uint, array_nd=1);
registerFunction('[pdarray]<uint8,uint64,1>', ark__pdarray__uint8_uint_1, 'IndexingMsg', 249);

proc ark__pdarray__uint8_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=uint(8), array_nd=1);
registerFunction('[pdarray]<uint8,uint8,1>', ark__pdarray__uint8_uint8_1, 'IndexingMsg', 249);

proc ark__pdarray__uint8_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=real, array_nd=1);
registerFunction('[pdarray]<uint8,float64,1>', ark__pdarray__uint8_real_1, 'IndexingMsg', 249);

proc ark__pdarray__uint8_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=bool, array_nd=1);
registerFunction('[pdarray]<uint8,bool,1>', ark__pdarray__uint8_bool_1, 'IndexingMsg', 249);

proc ark__pdarray__uint8_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_idx=bigint, array_nd=1);
registerFunction('[pdarray]<uint8,bigint,1>', ark__pdarray__uint8_bigint_1, 'IndexingMsg', 249);

proc ark__pdarray__real_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=int, array_nd=1);
registerFunction('[pdarray]<float64,int64,1>', ark__pdarray__real_int_1, 'IndexingMsg', 249);

proc ark__pdarray__real_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=uint, array_nd=1);
registerFunction('[pdarray]<float64,uint64,1>', ark__pdarray__real_uint_1, 'IndexingMsg', 249);

proc ark__pdarray__real_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=uint(8), array_nd=1);
registerFunction('[pdarray]<float64,uint8,1>', ark__pdarray__real_uint8_1, 'IndexingMsg', 249);

proc ark__pdarray__real_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=real, array_nd=1);
registerFunction('[pdarray]<float64,float64,1>', ark__pdarray__real_real_1, 'IndexingMsg', 249);

proc ark__pdarray__real_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=bool, array_nd=1);
registerFunction('[pdarray]<float64,bool,1>', ark__pdarray__real_bool_1, 'IndexingMsg', 249);

proc ark__pdarray__real_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=real, array_dtype_idx=bigint, array_nd=1);
registerFunction('[pdarray]<float64,bigint,1>', ark__pdarray__real_bigint_1, 'IndexingMsg', 249);

proc ark__pdarray__bool_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=int, array_nd=1);
registerFunction('[pdarray]<bool,int64,1>', ark__pdarray__bool_int_1, 'IndexingMsg', 249);

proc ark__pdarray__bool_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=uint, array_nd=1);
registerFunction('[pdarray]<bool,uint64,1>', ark__pdarray__bool_uint_1, 'IndexingMsg', 249);

proc ark__pdarray__bool_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=uint(8), array_nd=1);
registerFunction('[pdarray]<bool,uint8,1>', ark__pdarray__bool_uint8_1, 'IndexingMsg', 249);

proc ark__pdarray__bool_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=real, array_nd=1);
registerFunction('[pdarray]<bool,float64,1>', ark__pdarray__bool_real_1, 'IndexingMsg', 249);

proc ark__pdarray__bool_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=bool, array_nd=1);
registerFunction('[pdarray]<bool,bool,1>', ark__pdarray__bool_bool_1, 'IndexingMsg', 249);

proc ark__pdarray__bool_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_idx=bigint, array_nd=1);
registerFunction('[pdarray]<bool,bigint,1>', ark__pdarray__bool_bigint_1, 'IndexingMsg', 249);

proc ark__pdarray__bigint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=int, array_nd=1);
registerFunction('[pdarray]<bigint,int64,1>', ark__pdarray__bigint_int_1, 'IndexingMsg', 249);

proc ark__pdarray__bigint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=uint, array_nd=1);
registerFunction('[pdarray]<bigint,uint64,1>', ark__pdarray__bigint_uint_1, 'IndexingMsg', 249);

proc ark__pdarray__bigint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=uint(8), array_nd=1);
registerFunction('[pdarray]<bigint,uint8,1>', ark__pdarray__bigint_uint8_1, 'IndexingMsg', 249);

proc ark__pdarray__bigint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=real, array_nd=1);
registerFunction('[pdarray]<bigint,float64,1>', ark__pdarray__bigint_real_1, 'IndexingMsg', 249);

proc ark__pdarray__bigint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=bool, array_nd=1);
registerFunction('[pdarray]<bigint,bool,1>', ark__pdarray__bigint_bool_1, 'IndexingMsg', 249);

proc ark__pdarray__bigint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.multiPDArrayIndex(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_idx=bigint, array_nd=1);
registerFunction('[pdarray]<bigint,bigint,1>', ark__pdarray__bigint_bigint_1, 'IndexingMsg', 249);

proc ark__slice__pdarray_int_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=int, array_nd=1);
registerFunction('[slice]=pdarray<int64,int64,1>', ark__slice__pdarray_int_int_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_int_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=uint, array_nd=1);
registerFunction('[slice]=pdarray<int64,uint64,1>', ark__slice__pdarray_int_uint_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_int_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=uint(8), array_nd=1);
registerFunction('[slice]=pdarray<int64,uint8,1>', ark__slice__pdarray_int_uint8_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_int_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=real, array_nd=1);
registerFunction('[slice]=pdarray<int64,float64,1>', ark__slice__pdarray_int_real_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_int_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=bool, array_nd=1);
registerFunction('[slice]=pdarray<int64,bool,1>', ark__slice__pdarray_int_bool_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_int_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=int, array_dtype_b=bigint, array_nd=1);
registerFunction('[slice]=pdarray<int64,bigint,1>', ark__slice__pdarray_int_bigint_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_uint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=int, array_nd=1);
registerFunction('[slice]=pdarray<uint64,int64,1>', ark__slice__pdarray_uint_int_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_uint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=uint, array_nd=1);
registerFunction('[slice]=pdarray<uint64,uint64,1>', ark__slice__pdarray_uint_uint_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_uint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=uint(8), array_nd=1);
registerFunction('[slice]=pdarray<uint64,uint8,1>', ark__slice__pdarray_uint_uint8_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_uint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=real, array_nd=1);
registerFunction('[slice]=pdarray<uint64,float64,1>', ark__slice__pdarray_uint_real_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_uint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=bool, array_nd=1);
registerFunction('[slice]=pdarray<uint64,bool,1>', ark__slice__pdarray_uint_bool_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_uint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint, array_dtype_b=bigint, array_nd=1);
registerFunction('[slice]=pdarray<uint64,bigint,1>', ark__slice__pdarray_uint_bigint_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_uint8_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=int, array_nd=1);
registerFunction('[slice]=pdarray<uint8,int64,1>', ark__slice__pdarray_uint8_int_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_uint8_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=uint, array_nd=1);
registerFunction('[slice]=pdarray<uint8,uint64,1>', ark__slice__pdarray_uint8_uint_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_uint8_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=uint(8), array_nd=1);
registerFunction('[slice]=pdarray<uint8,uint8,1>', ark__slice__pdarray_uint8_uint8_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_uint8_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=real, array_nd=1);
registerFunction('[slice]=pdarray<uint8,float64,1>', ark__slice__pdarray_uint8_real_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_uint8_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=bool, array_nd=1);
registerFunction('[slice]=pdarray<uint8,bool,1>', ark__slice__pdarray_uint8_bool_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_uint8_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=uint(8), array_dtype_b=bigint, array_nd=1);
registerFunction('[slice]=pdarray<uint8,bigint,1>', ark__slice__pdarray_uint8_bigint_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_real_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=int, array_nd=1);
registerFunction('[slice]=pdarray<float64,int64,1>', ark__slice__pdarray_real_int_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_real_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=uint, array_nd=1);
registerFunction('[slice]=pdarray<float64,uint64,1>', ark__slice__pdarray_real_uint_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_real_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=uint(8), array_nd=1);
registerFunction('[slice]=pdarray<float64,uint8,1>', ark__slice__pdarray_real_uint8_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_real_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=real, array_nd=1);
registerFunction('[slice]=pdarray<float64,float64,1>', ark__slice__pdarray_real_real_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_real_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=bool, array_nd=1);
registerFunction('[slice]=pdarray<float64,bool,1>', ark__slice__pdarray_real_bool_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_real_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=real, array_dtype_b=bigint, array_nd=1);
registerFunction('[slice]=pdarray<float64,bigint,1>', ark__slice__pdarray_real_bigint_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_bool_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=int, array_nd=1);
registerFunction('[slice]=pdarray<bool,int64,1>', ark__slice__pdarray_bool_int_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_bool_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=uint, array_nd=1);
registerFunction('[slice]=pdarray<bool,uint64,1>', ark__slice__pdarray_bool_uint_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_bool_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=uint(8), array_nd=1);
registerFunction('[slice]=pdarray<bool,uint8,1>', ark__slice__pdarray_bool_uint8_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_bool_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=real, array_nd=1);
registerFunction('[slice]=pdarray<bool,float64,1>', ark__slice__pdarray_bool_real_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_bool_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=bool, array_nd=1);
registerFunction('[slice]=pdarray<bool,bool,1>', ark__slice__pdarray_bool_bool_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_bool_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bool, array_dtype_b=bigint, array_nd=1);
registerFunction('[slice]=pdarray<bool,bigint,1>', ark__slice__pdarray_bool_bigint_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_bigint_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=int, array_nd=1);
registerFunction('[slice]=pdarray<bigint,int64,1>', ark__slice__pdarray_bigint_int_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_bigint_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=uint, array_nd=1);
registerFunction('[slice]=pdarray<bigint,uint64,1>', ark__slice__pdarray_bigint_uint_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_bigint_uint8_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=uint(8), array_nd=1);
registerFunction('[slice]=pdarray<bigint,uint8,1>', ark__slice__pdarray_bigint_uint8_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_bigint_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=real, array_nd=1);
registerFunction('[slice]=pdarray<bigint,float64,1>', ark__slice__pdarray_bigint_real_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_bigint_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=bool, array_nd=1);
registerFunction('[slice]=pdarray<bigint,bool,1>', ark__slice__pdarray_bigint_bool_1, 'IndexingMsg', 940);

proc ark__slice__pdarray_bigint_bigint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return IndexingMsg.setSliceIndexToPdarray(cmd, msgArgs, st, array_dtype_a=bigint, array_dtype_b=bigint, array_nd=1);
registerFunction('[slice]=pdarray<bigint,bigint,1>', ark__slice__pdarray_bigint_bigint_1, 'IndexingMsg', 940);

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
