module Commands {

use CommandMap, Message, MultiTypeSymbolTable, MultiTypeSymEntry;

import ManipulationMsg;

proc ark_broadcast_int_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<int64,1,1>', ark_broadcast_int_1_1, 'ManipulationMsg', 62);

proc ark_broadcast_int_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=2);
registerFunction('broadcast<int64,1,2>', ark_broadcast_int_1_2, 'ManipulationMsg', 62);

proc ark_broadcast_int_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=1, array_nd_out=3);
registerFunction('broadcast<int64,1,3>', ark_broadcast_int_1_3, 'ManipulationMsg', 62);

proc ark_broadcast_int_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=1);
registerFunction('broadcast<int64,2,1>', ark_broadcast_int_2_1, 'ManipulationMsg', 62);

proc ark_broadcast_int_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=2);
registerFunction('broadcast<int64,2,2>', ark_broadcast_int_2_2, 'ManipulationMsg', 62);

proc ark_broadcast_int_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=2, array_nd_out=3);
registerFunction('broadcast<int64,2,3>', ark_broadcast_int_2_3, 'ManipulationMsg', 62);

proc ark_broadcast_int_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=1);
registerFunction('broadcast<int64,3,1>', ark_broadcast_int_3_1, 'ManipulationMsg', 62);

proc ark_broadcast_int_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=2);
registerFunction('broadcast<int64,3,2>', ark_broadcast_int_3_2, 'ManipulationMsg', 62);

proc ark_broadcast_int_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=int, array_nd_in=3, array_nd_out=3);
registerFunction('broadcast<int64,3,3>', ark_broadcast_int_3_3, 'ManipulationMsg', 62);

proc ark_broadcast_real_1_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=1);
registerFunction('broadcast<float64,1,1>', ark_broadcast_real_1_1, 'ManipulationMsg', 62);

proc ark_broadcast_real_1_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=2);
registerFunction('broadcast<float64,1,2>', ark_broadcast_real_1_2, 'ManipulationMsg', 62);

proc ark_broadcast_real_1_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=1, array_nd_out=3);
registerFunction('broadcast<float64,1,3>', ark_broadcast_real_1_3, 'ManipulationMsg', 62);

proc ark_broadcast_real_2_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=1);
registerFunction('broadcast<float64,2,1>', ark_broadcast_real_2_1, 'ManipulationMsg', 62);

proc ark_broadcast_real_2_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=2);
registerFunction('broadcast<float64,2,2>', ark_broadcast_real_2_2, 'ManipulationMsg', 62);

proc ark_broadcast_real_2_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=2, array_nd_out=3);
registerFunction('broadcast<float64,2,3>', ark_broadcast_real_2_3, 'ManipulationMsg', 62);

proc ark_broadcast_real_3_1(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=1);
registerFunction('broadcast<float64,3,1>', ark_broadcast_real_3_1, 'ManipulationMsg', 62);

proc ark_broadcast_real_3_2(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=2);
registerFunction('broadcast<float64,3,2>', ark_broadcast_real_3_2, 'ManipulationMsg', 62);

proc ark_broadcast_real_3_3(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
  return ManipulationMsg.broadcastToMsg(cmd, msgArgs, st, array_dtype=real, array_nd_in=3, array_nd_out=3);
registerFunction('broadcast<float64,3,3>', ark_broadcast_real_3_3, 'ManipulationMsg', 62);

}