
module ServerErrorStrings
{
    use ServerConfig;
    use MultiTypeSymEntry;

    // generate consistent error strings for notImplementedError
    proc notImplementedError(pname: string, ldtype: DType, op: string, rdtype: DType):string {
        return try! "Error: %s: %s %s %s not implemented".format(pname,
                                                                 dtype2str(ldtype),
                                                                 op,
                                                                 dtype2str(rdtype));
    }
    proc notImplementedError(pname: string, efunc: string, ldtype: DType):string {
        return try! "Error: %s: %s %s not implemented".format(pname,
                                                              efunc,
                                                              dtype2str(ldtype));
    }
    proc notImplementedError(pname: string, dtype: DType):string {
        return try! "Error: %s: %s not implemented".format(pname,
                                                           dtype2str(dtype));
    }
    proc notImplementedError(pname: string, arg: string):string {
        return try! "Error: %s: %s not implemented".format(pname,arg);
    }

    // generate consistent error strings unrecognizedTypeError
    proc unrecognizedTypeError(pname: string, stype: string):string {
        return try! "Error: %s: unrecognized type: %s".format(pname, stype);
    }

    // generate consistent error strings unknownSymbolError
    proc unknownSymbolError(pname: string, sname: string):string {
        return try! "Error: %s: unkown symbol: %s".format(pname, sname);
    }

}

