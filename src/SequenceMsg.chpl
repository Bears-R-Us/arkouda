module SequenceMsg {
    use ServerConfig;

    use Time;
    use Reflection;
    use Logging;
    use Message;
    use BigInteger;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    
    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const smLogger = new Logger(logLevel, logChannel);
    /*
    Creates a sym entry with distributed array adhering to the Msg parameters (start, stop, step)

    :arg reqMsg: request containing (cmd,start,stop,step)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: MsgTuple
    */
    @arkouda.instantiateAndRegister
    proc arange(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws 
        where ((array_dtype==int) || (array_dtype==uint(64)) || (array_dtype==bigint)) && (array_nd==1) {
        
        const start =  msgArgs["start"].toScalar(array_dtype),
            stop =  msgArgs["stop"].toScalar(array_dtype),
            step =  msgArgs["step"].toScalar(array_dtype),
            len = ((stop - start + step - 1) / step):int;

        var ea = makeDistArray(len, array_dtype);
        const ref ead = ea.domain;
        forall (ei, i) in zip(ea, ead) {
            ei = start + (i * step);
        }
        return st.insert(new shared SymEntry(ea));
    }

    //  The following functions implement a linspace, including multi-dim start and stop, and the
    //  endpoint flag.  The axis variable is handled python-side.

    //  There are functions noted as _ss, _sv, _vs, and _vv.  The first letter indicates whether
    //  start is a scalar or vector, and the second letter has the same meaning for stop.

    //  ss means both inputs are scalar.

    @arkouda.registerCommand()
    proc linspace_ss(start : real, stop : real, num : int, endpoint: bool) : [] real throws {

        var divisor = if endpoint then num - 1 else num;
        var delta = (stop - start) / divisor;
        overMemLimit(8*num);

        var result : [0..#num] real;
        for i in 0..#num {
            result[i] = start + i*delta ;
        }

        return result;
    }

    // This proc returns a shape that is an_int prepended to a_shape.
    // e.g., if a_shape is (2,3) and an_int is 4, it will return (4,2,3).
    // This isn't used by linspace_ss, of course, but the other 3 procs all use it.

    proc linspace_shape(a_shape : ?N*int, an_int : int) : (N + 1) * int {
        var shapeOut : (N+1)*int ;
        shapeOut[0] = an_int;
        for i in 0..<N do shapeOut[i+1] = a_shape[i];
        return shapeOut;
    }

    //  sv means start is scalar, and stop is vector.

    @arkouda.registerCommand()
    proc linspace_sv(start : real, stop : [?d] real, num : int, endpoint: bool) : [] real throws {

        var divisor = if endpoint then num - 1 else num;
        var delta : [d] real;
        for (del,sta) in zip (delta,stop) {
            del = (sta - start) / divisor;
        }
        overMemLimit(8*num*stop.size);

        var result = makeDistArray((...linspace_shape(stop.shape,num)),real);
        if d.rank == 1 {    // if rank of stop is 1, then
            for idx in d {  // idx is an integer.
                for j in 0..#num {
                    result[j,idx] = start + j*delta[idx] ;
                }
            }
        } else { // if rank of stop is > 1, we have to create an index into r.
            var rdx : (d.rank+1)*int;
            for idx in d {
                for i in 0..<d.rank do rdx[i+1] = idx[i]; // TODO: it may be possible to do this with slices
                for j in 0..#num {
                    rdx[0] = j;
                    result[rdx] = start + j*delta[idx] ;
                }
            }
        }
        return result;
    }

    //  vs means start is vector, and stop is scalar.

    @arkouda.registerCommand()
    proc linspace_vs(start : [?d] real, stop : real, num : int, endpoint: bool) : [] real throws {

        var divisor = if endpoint then num - 1 else num;
        var delta : [d] real;
        for (del,sta) in zip (delta,start) {
            del = (stop - sta) / divisor;
        }
        overMemLimit(8*num*start.size);

        var result = makeDistArray((...linspace_shape(start.shape,num)),real);
        if d.rank == 1 {    // if rank of start is 1, then
            for idx in d {  // idx is an integer.
                for j in 0..#num {
                    result[j,idx] = start[idx] + j*delta[idx] ;
                }
            }
        } else { // if rank of start is > 1, we have to create an index into r.
            var rdx : (d.rank+1)*int;
            for idx in d {
                for i in 0..<d.rank do rdx[i+1] = idx[i]; // TODO: it may be possible to do this with slices
                for j in 0..#num {
                    rdx[0] = j;
                    result[rdx] = start[idx] + j*delta[idx] ;
                }
            }
        }
        return result;
    }

    //  in the vv (vector-vector) case, both start and stop will have already been broadcast to a
    //  compatible shape before the chapel code is invoked.

    @arkouda.registerCommand()
    proc linspace_vv(start : [?d] real, stop : [d] real, num : int, endpoint: bool) : [] real throws {

        var divisor = if endpoint then num - 1 else num;
        var delta : [d] real;
        for (del,sta, sto) in zip (delta,start,stop) {
            del = (sto - sta) / divisor;
        }
        overMemLimit(8*num*start.size);

        var result = makeDistArray((...linspace_shape(start.shape,num)),real);
        if d.rank == 1 {    // if rank of start and stop is 1, then
            for idx in d {  // idx is an integer.
                for j in 0..#num {
                    result[j,idx] = start[idx] + j*delta[idx] ;
                }
            }
        } else { // if rank of start and stop is > 1, we have to create an index into r.
            var rdx : (d.rank+1)*int;
            for idx in d {
                for i in 0..<d.rank do rdx[i+1] = idx[i]; // TODO: it may be possible to do this with slices
                for j in 0..#num {
                    rdx[0] = j;
                    result[rdx] = start[idx] + j*delta[idx] ;
                }
            }
        }
        return result;
    }


}
