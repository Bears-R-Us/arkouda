/* error string to return to client... generate consistent error strings for notImplementedError */
module ServerErrorStrings
{
    use NumPyDType;
    use IO;
    
    class ErrorWithMsg: Error {
      var msg: string;
    }

    /* binary operator is not implemented for DTypes */
    proc notImplementedError(pname: string, ldtype: DType, op: string, rdtype: DType):string {
        return try! "Error: %s: %s %s %s not implemented".format(pname,
                                                                 dtype2str(ldtype),
                                                                 op,
                                                                 dtype2str(rdtype));
    }
    proc notImplementedError(pname: string, ldtype: string, op: string, rdtype: string):string {
        return try! "Error: %s: %s %s %s not implemented".format(pname,
                                                                 ldtype,
                                                                 op,
                                                                 rdtype);
    }
    /* efunc is not implemented for DType */
    proc notImplementedError(pname: string, efunc: string, ldtype: DType):string {
        return try! "Error: %s: %s %s not implemented".format(pname,
                                                              efunc,
                                                              dtype2str(ldtype));
    }
    /* efunc is not implemented for DTypes */
    proc notImplementedError(pname: string, efunc: string, dt1: DType, dt2: DType, dt3: DType): string {
      return try! "Error: %s: %s %s %s %s not implemented".format(pname,
                                                                  efunc,
                                                                  dtype2str(dt1),
                                                                  dtype2str(dt2),
                                                                  dtype2str(dt3));
    }
    /* algorthm is not implemented for DType */
    proc notImplementedError(pname: string, dtype: DType):string {
        return try! "Error: %s: %s not implemented".format(pname,
                                                           dtype2str(dtype));
    }
    /* proc is not implemented for this kind of argument */
    proc notImplementedError(pname: string, arg: string):string {
        return try! "Error: %s: %s not implemented".format(pname,arg);
    }

    /* unrecognized DType */
    proc unrecognizedTypeError(pname: string, stype: string):string {
        return try! "Error: %s: unrecognized type: %s".format(pname, stype);
    }

    /* name not found in the symbol table */
    proc unknownSymbolError(pname: string, sname: string):string {
      if pname != "" then
        return try! "Error: %s: unknown symbol: %s".format(pname, sname);
      else
        return try! "Error: unknown symbol: %s".format(sname);
    }

    proc unknownError(pname: string): string {
      return try! "Error: %s: unexpected error".format(pname);
    }

    /* incompatible arguments */
    proc incompatibleArgumentsError(pname: string, reason: string) {
      return try! "Error: %s: Incompatible arguments: %s".format(pname, reason);
    }

    
}

