module Commands {

use CommandMap, Message, MultiTypeSymbolTable, MultiTypeSymEntry;

import ArgSortMsg;

import ArraySetopsMsg;

import BroadcastMsg;

import CastMsg;

import ConcatenateMsg;

import CSVMsg;

import DataFrameIndexingMsg;

import EfuncMsg;

import EncodingMsg;

import FlattenMsg;

import HashMsg;

import HDF5Msg;

import HistogramMsg;

import In1dMsg;

import IndexingMsg;

import JoinEqWithDTMsg;

import KExtremeMsg;

import LogMsg;

import ManipulationMsg;

proc ark_broadcast_int_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<int64,1,1>', ark_broadcast_int_1_1, 'ManipulationMsg', 62);

proc ark_broadcast_uint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<uint64,1,1>', ark_broadcast_uint_1_1, 'ManipulationMsg', 62);

proc ark_broadcast_real_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<float64,1,1>', ark_broadcast_real_1_1, 'ManipulationMsg', 62);

proc ark_broadcast_bool_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<bool,1,1>', ark_broadcast_bool_1_1, 'ManipulationMsg', 62);

proc ark_concat_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('concat<int64,1>', ark_concat_int_1, 'ManipulationMsg', 159);

proc ark_concat_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('concat<uint64,1>', ark_concat_uint_1, 'ManipulationMsg', 159);

proc ark_concat_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('concat<float64,1>', ark_concat_real_1, 'ManipulationMsg', 159);

proc ark_concat_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('concat<bool,1>', ark_concat_bool_1, 'ManipulationMsg', 159);

proc ark_concatFlat_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('concatFlat<int64,1>', ark_concatFlat_int_1, 'ManipulationMsg', 215);

proc ark_concatFlat_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('concatFlat<uint64,1>', ark_concatFlat_uint_1, 'ManipulationMsg', 215);

proc ark_concatFlat_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('concatFlat<float64,1>', ark_concatFlat_real_1, 'ManipulationMsg', 215);

proc ark_concatFlat_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.concatFlatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('concatFlat<bool,1>', ark_concatFlat_bool_1, 'ManipulationMsg', 215);

proc ark_expandDims_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('expandDims<int64,1>', ark_expandDims_int_1, 'ManipulationMsg', 239);

proc ark_expandDims_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('expandDims<uint64,1>', ark_expandDims_uint_1, 'ManipulationMsg', 239);

proc ark_expandDims_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('expandDims<float64,1>', ark_expandDims_real_1, 'ManipulationMsg', 239);

proc ark_expandDims_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.expandDimsMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('expandDims<bool,1>', ark_expandDims_bool_1, 'ManipulationMsg', 239);

proc ark_flip_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('flip<int64,1>', ark_flip_int_1, 'ManipulationMsg', 291);

proc ark_flip_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('flip<uint64,1>', ark_flip_uint_1, 'ManipulationMsg', 291);

proc ark_flip_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('flip<float64,1>', ark_flip_real_1, 'ManipulationMsg', 291);

proc ark_flip_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('flip<bool,1>', ark_flip_bool_1, 'ManipulationMsg', 291);

proc ark_flipAll_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('flipAll<int64,1>', ark_flipAll_int_1, 'ManipulationMsg', 359);

proc ark_flipAll_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('flipAll<uint64,1>', ark_flipAll_uint_1, 'ManipulationMsg', 359);

proc ark_flipAll_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('flipAll<float64,1>', ark_flipAll_real_1, 'ManipulationMsg', 359);

proc ark_flipAll_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.flipAllMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('flipAll<bool,1>', ark_flipAll_bool_1, 'ManipulationMsg', 359);

proc ark_permute_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('permute<int64,1>', ark_permute_int_1, 'ManipulationMsg', 390);

proc ark_permute_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('permute<uint64,1>', ark_permute_uint_1, 'ManipulationMsg', 390);

proc ark_permute_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('permute<float64,1>', ark_permute_real_1, 'ManipulationMsg', 390);

proc ark_permute_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.permuteDims(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('permute<bool,1>', ark_permute_bool_1, 'ManipulationMsg', 390);

proc ark_reshape_int_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<int64,1,1>', ark_reshape_int_1_1, 'ManipulationMsg', 440);

proc ark_reshape_uint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<uint64,1,1>', ark_reshape_uint_1_1, 'ManipulationMsg', 440);

proc ark_reshape_real_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<float64,1,1>', ark_reshape_real_1_1, 'ManipulationMsg', 440);

proc ark_reshape_bool_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.reshapeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=1);
registerFunction('reshape<bool,1,1>', ark_reshape_bool_1_1, 'ManipulationMsg', 440);

proc ark_roll_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('roll<int64,1>', ark_roll_int_1, 'ManipulationMsg', 522);

proc ark_roll_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('roll<uint64,1>', ark_roll_uint_1, 'ManipulationMsg', 522);

proc ark_roll_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('roll<float64,1>', ark_roll_real_1, 'ManipulationMsg', 522);

proc ark_roll_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('roll<bool,1>', ark_roll_bool_1, 'ManipulationMsg', 522);

proc ark_rollFlattened_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('rollFlattened<int64,1>', ark_rollFlattened_int_1, 'ManipulationMsg', 587);

proc ark_rollFlattened_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('rollFlattened<uint64,1>', ark_rollFlattened_uint_1, 'ManipulationMsg', 587);

proc ark_rollFlattened_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('rollFlattened<float64,1>', ark_rollFlattened_real_1, 'ManipulationMsg', 587);

proc ark_rollFlattened_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.rollFlattenedMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('rollFlattened<bool,1>', ark_rollFlattened_bool_1, 'ManipulationMsg', 587);

proc ark_squeeze_int_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<int64,1,1>', ark_squeeze_int_1_1, 'ManipulationMsg', 608);

proc ark_squeeze_uint_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=uint, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<uint64,1,1>', ark_squeeze_uint_1_1, 'ManipulationMsg', 608);

proc ark_squeeze_real_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<float64,1,1>', ark_squeeze_real_1_1, 'ManipulationMsg', 608);

proc ark_squeeze_bool_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.squeezeMsg(cmd, msgArgs, st, array_dtype=bool, array_nd_in=1, array_nd_out=1);
registerFunction('squeeze<bool,1,1>', ark_squeeze_bool_1_1, 'ManipulationMsg', 608);

proc ark_stack_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('stack<int64,1>', ark_stack_int_1, 'ManipulationMsg', 687);

proc ark_stack_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('stack<uint64,1>', ark_stack_uint_1, 'ManipulationMsg', 687);

proc ark_stack_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('stack<float64,1>', ark_stack_real_1, 'ManipulationMsg', 687);

proc ark_stack_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.stackMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('stack<bool,1>', ark_stack_bool_1, 'ManipulationMsg', 687);

proc ark_tile_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('tile<int64,1>', ark_tile_int_1, 'ManipulationMsg', 778);

proc ark_tile_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('tile<uint64,1>', ark_tile_uint_1, 'ManipulationMsg', 778);

proc ark_tile_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('tile<float64,1>', ark_tile_real_1, 'ManipulationMsg', 778);

proc ark_tile_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.tileMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('tile<bool,1>', ark_tile_bool_1, 'ManipulationMsg', 778);

proc ark_unstack_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('unstack<int64,1>', ark_unstack_int_1, 'ManipulationMsg', 819);

proc ark_unstack_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('unstack<uint64,1>', ark_unstack_uint_1, 'ManipulationMsg', 819);

proc ark_unstack_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('unstack<float64,1>', ark_unstack_real_1, 'ManipulationMsg', 819);

proc ark_unstack_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.unstackMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('unstack<bool,1>', ark_unstack_bool_1, 'ManipulationMsg', 819);

proc ark_repeatFlat_int_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=int, array_nd=1);
registerFunction('repeatFlat<int64,1>', ark_repeatFlat_int_1, 'ManipulationMsg', 903);

proc ark_repeatFlat_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=uint, array_nd=1);
registerFunction('repeatFlat<uint64,1>', ark_repeatFlat_uint_1, 'ManipulationMsg', 903);

proc ark_repeatFlat_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=real, array_nd=1);
registerFunction('repeatFlat<float64,1>', ark_repeatFlat_real_1, 'ManipulationMsg', 903);

proc ark_repeatFlat_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.repeatFlatMsg(cmd, msgArgs, st, array_dtype=bool, array_nd=1);
registerFunction('repeatFlat<bool,1>', ark_repeatFlat_bool_1, 'ManipulationMsg', 903);

import OperatorMsg;

import ParquetMsg;

import RandMsg;

import ReductionMsg;

import RegistrationMsg;

import SegmentedMsg;

import SequenceMsg;

import SortMsg;

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
registerFunction('mean<int64,1>', ark_mean_int_1, 'StatsMsg', 24);

proc ark_mean_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('mean<uint64,1>', ark_mean_uint_1, 'StatsMsg', 24);

proc ark_mean_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('mean<float64,1>', ark_mean_real_1, 'StatsMsg', 24);

proc ark_mean_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_mean_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('mean<bool,1>', ark_mean_bool_1, 'StatsMsg', 24);

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
registerFunction('meanReduce<int64,1>', ark_meanReduce_int_1, 'StatsMsg', 31);

proc ark_meanReduce_uint_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=uint, array_nd_0=1);
registerFunction('meanReduce<uint64,1>', ark_meanReduce_uint_1, 'StatsMsg', 31);

proc ark_meanReduce_real_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=real, array_nd_0=1);
registerFunction('meanReduce<float64,1>', ark_meanReduce_real_1, 'StatsMsg', 31);

proc ark_meanReduce_bool_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ark_reg_meanReduce_generic(cmd, msgArgs, st, array_dtype_0=bool, array_nd_0=1);
registerFunction('meanReduce<bool,1>', ark_meanReduce_bool_1, 'StatsMsg', 31);

import TimeClassMsg;

import TransferMsg;

import UniqueMsg;

}