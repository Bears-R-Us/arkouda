use CommandMap, Message, MultiTypeSymbolTable;

    proc arkouda_nd_gen_argsortMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return argsortMsg(cmd, msgArgs, st, 1);

    registerFunction("argsort1D", arkouda_nd_gen_argsortMsg1D);
    use ArgSortMsg;

    proc arkouda_nd_gen_binopvvMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return binopvvMsg(cmd, msgArgs, st, 1);

    registerFunction("binopvv1D", arkouda_nd_gen_binopvvMsg1D);
    
    proc arkouda_nd_gen_binopvsMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return binopvsMsg(cmd, msgArgs, st, 1);

    registerFunction("binopvs1D", arkouda_nd_gen_binopvsMsg1D);
    
    proc arkouda_nd_gen_binopsvMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return binopsvMsg(cmd, msgArgs, st, 1);

    registerFunction("binopsv1D", arkouda_nd_gen_binopsvMsg1D);
    
    proc arkouda_nd_gen_opeqvvMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return opeqvvMsg(cmd, msgArgs, st, 1);

    registerFunction("opeqvv1D", arkouda_nd_gen_opeqvvMsg1D);
    
    proc arkouda_nd_gen_opeqvsMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return opeqvsMsg(cmd, msgArgs, st, 1);

    registerFunction("opeqvs1D", arkouda_nd_gen_opeqvsMsg1D);
    use OperatorMsg;

    proc arkouda_nd_gen_randintMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return randintMsg(cmd, msgArgs, st, 1);

    registerFunction("randint1D", arkouda_nd_gen_randintMsg1D);
    use RandMsg;

    proc arkouda_nd_gen_intIndexMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return intIndexMsg(cmd, msgArgs, st, 1);

    registerFunction("[int]1D", arkouda_nd_gen_intIndexMsg1D);
    
    proc arkouda_nd_gen_sliceIndexMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return sliceIndexMsg(cmd, msgArgs, st, 1);

    registerFunction("[slice]1D", arkouda_nd_gen_sliceIndexMsg1D);
    
    proc arkouda_nd_gen_takeAlongAxisMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return takeAlongAxisMsg(cmd, msgArgs, st, 1);

    registerFunction("takeAlongAxis1D", arkouda_nd_gen_takeAlongAxisMsg1D);
    use IndexingMsg;

    proc arkouda_nd_gen_reductionMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return reductionMsg(cmd, msgArgs, st, 1);

    registerFunction("reduction1D", arkouda_nd_gen_reductionMsg1D);
    use ReductionMsg;

    proc arkouda_nd_gen_efuncMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return efuncMsg(cmd, msgArgs, st, 1);

    registerFunction("efunc1D", arkouda_nd_gen_efuncMsg1D);
    use EfuncMsg;

    proc arkouda_nd_gen_createMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return createMsg(cmd, msgArgs, st, 1);

    registerFunction("create1D", arkouda_nd_gen_createMsg1D);
    
    proc arkouda_nd_gen_setMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return setMsg(cmd, msgArgs, st, 1);

    registerFunction("set1D", arkouda_nd_gen_setMsg1D);
    use MsgProcessing;

    proc _nd_gen_tondarrayMsg1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): bytes throws
        do return tondarrayMsg(cmd, msgArgs, st, 1);

    registerBinaryFunction("tondarray1D", _nd_gen_tondarrayMsg1D);
    use GenSymIO;

    proc _nd_gen_broadcast1Dx1D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return broadcastNDArray(cmd, msgArgs, st, 1, 1);

    registerFunction("broadcast1Dx1D", _nd_gen_broadcast1Dx1D);
    