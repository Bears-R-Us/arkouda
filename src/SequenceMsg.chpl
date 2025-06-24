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

    //  The following functions implement linspace, including multi-dim start and stop, and the
    //  endpoint flag.  The axis variable is handled python-side.

    //  Either both inputs are scalars, or both are vectors.  The scalar-vector and vector-scalar
    //  cases are converted to vector-vector python-side.

    //  ss means both inputs are scalar.

    @arkouda.registerCommand()
    proc linspace_ss(start : real, stop : real, num : int, endpoint: bool) : [] real throws {

        var divisor = if endpoint then num - 1 else num;
        var delta = (stop - start) / divisor;
        overMemLimit(8*num);

        var result = makeDistArray(num,real);
        forall i in 0..#num {
            result[i] = start + i*delta ;
        }

        return result;
    }

    // This proc returns a shape that is an_int prepended to a_shape.
    // e.g., if a_shape is (2,3) and an_int is 4, it will return (4,2,3).

    proc linspace_shape(a_shape : ?N*int, an_int : int) : (N + 1) * int {
        var shapeOut : (N+1)*int ;
        shapeOut[0] = an_int;
        for i in 0..<N do shapeOut[i+1] = a_shape[i];
        return shapeOut;
    }

    //  in the vv (vector-vector) case, both start and stop will have already been broadcast to a
    //  compatible shape before the chapel code is invoked.

    @arkouda.registerCommand()
    proc linspace_vv(start : [?d] real, stop : [d] real, num : int, endpoint: bool) : [] real throws {

        var divisor = if endpoint then num - 1 else num;
        var delta : [d] real;
        forall (del,sta, stp) in zip (delta,start,stop) {
            del = (stp - sta) / divisor;
        }
        overMemLimit(8*num*start.size);

        var result = makeDistArray((...linspace_shape(start.shape,num)),real);
        if d.rank == 1 {    // if rank of start and stop is 1, then
            for idx in d {  // idx is an integer.
                for j in 0..#num {
                    result[j,idx] = start[idx] + j*delta[idx] ;
                }
            }
        } else { // if rank of start and stop is > 1, we have to create an index into result.
            forall j in 0..#num {
                for idx in d {
                    const rdx = ((j),(...idx)); // prepends j to the tuple idx
                    result[rdx] = start[idx] + j*delta[idx];
                }
            }
        }
        return result;
    }


}
