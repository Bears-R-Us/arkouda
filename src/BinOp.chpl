
module BinOp
{
  use ServerConfig;
  
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Logging;
  use Message;
  use BitOps;
  use BigInteger;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const omLogger = new Logger(logLevel, logChannel);

  /*
  Helper function to ensure that floor division cases are handled in accordance with numpy
  */
  inline proc floorDivisionHelper(numerator: ?t, denom: ?t2): real {
    if (numerator == 0 && denom == 0) || (isinf(numerator) && (denom != 0 || isinf(denom))){
      return NAN;
    }
    else if (numerator > 0 && denom == -INFINITY) || (numerator < 0 && denom == INFINITY){
      return -1:real;
    }
    else {
      return floor(numerator/denom);
    }
  }

  /*
  Generic function to execute a binary operation on pdarray entries 
  in the symbol table

  :arg l: symbol table entry of the LHS operand

  :arg r: symbol table entry of the RHS operand

  :arg e: symbol table entry to store result of operation

  :arg op: string representation of binary operation to execute
  :type op: string

  :arg rname: name of the `e` in the symbol table
  :type rname: string

  :arg pn: routine name of callsite function
  :type pn: string

  :arg st: SymTab to act on
  :type st: borrowed SymTab 

  :returns: (MsgTuple) 
  :throws: `UndefinedSymbolError(name)`
  */
  proc doBinOpvv(l, r, e, op: string, rname, pn, st) throws {
    if e.etype == bool {
      // Since we know that the result type is a boolean, we know
      // that it either (1) is an operation between bools or (2) uses
      // a boolean operator (<, <=, etc.)
      if l.etype == bool && r.etype == bool {
        select op {
          when "|" {
            e.a = l.a | r.a;
          }
          when "&" {
            e.a = l.a & r.a;
          }
          when "^" {
            e.a = l.a ^ r.a;
          }
          when "==" {
            e.a = l.a == r.a;
          }
          when "!=" {
            e.a = l.a != r.a;
          }
          when "<" {
            e.a = l.a:int < r.a:int;
          }
          when ">" {
            e.a = l.a:int > r.a:int;
          }
          when "<=" {
            e.a = l.a:int <= r.a:int;
          }
          when ">=" {
            e.a = l.a:int >= r.a:int;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      }
      // All types support the same binary operations when the resultant
      // type is bool and `l` and `r` are not both boolean, so this does
      // not need to be specialized for each case.
      else {
        if ((l.etype == real && r.etype == bool) || (l.etype == bool && r.etype == real)) {
          select op {
            when "<" {
              e.a = l.a:real < r.a:real;
            }
            when ">" {
              e.a = l.a:real > r.a:real;
            }
            when "<=" {
              e.a = l.a:real <= r.a:real;
            }
            when ">=" {
              e.a = l.a:real >= r.a:real;
            }
            when "==" {
              e.a = l.a:real == r.a:real;
            }
            when "!=" {
              e.a = l.a:real != r.a:real;
            }
            otherwise {
              var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
              omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
            }
          }
        }
        else {
          select op {
            when "<" {
              e.a = l.a < r.a;
            }
            when ">" {
              e.a = l.a > r.a;
            }
            when "<=" {
              e.a = l.a <= r.a;
            }
            when ">=" {
              e.a = l.a >= r.a;
            }
            when "==" {
              e.a = l.a == r.a;
            }
            when "!=" {
              e.a = l.a != r.a;
            }
            otherwise {
              var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
              omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                              
              return new MsgTuple(errorMsg, MsgType.ERROR); 
            }
          }
        }
      }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    // Since we know that both `l` and `r` are of type `int` and that
    // the resultant type is not bool (checked in first `if`), we know
    // what operations are supported based on the resultant type
    else if (l.etype == int && r.etype == int) ||
            (l.etype == uint && r.etype == uint)  {
      if e.etype == int || e.etype == uint {
        select op {
          when "+" {
            e.a = l.a + r.a;
          }
          when "-" {
            e.a = l.a - r.a;
          }
          when "*" {
            e.a = l.a * r.a;
          }
          when "//" { // floordiv
            ref ea = e.a;
            ref la = l.a;
            ref ra = r.a;
            [(ei,li,ri) in zip(ea,la,ra)] ei = if ri != 0 then li/ri else 0;
          }
          when "%" { // modulo
            ref ea = e.a;
            ref la = l.a;
            ref ra = r.a;
            [(ei,li,ri) in zip(ea,la,ra)] ei = if ri != 0 then li%ri else 0;
          }
          when "<<" {
            e.a = l.a << r.a;
          }                    
          when ">>" {
            e.a = l.a >> r.a;
          }
          when "<<<" {
            e.a = rotl(l.a, r.a);
          }
          when ">>>" {
            e.a = rotr(l.a, r.a);
          }
          when "&" {
            e.a = l.a & r.a;
          }                    
          when "|" {
            e.a = l.a | r.a;
          }                    
          when "^" {
            e.a = l.a ^ r.a;
          }
          when "**" { 
            if || reduce (r.a<0){
              //instead of error, could we paste the below code but of type float?
              var errorMsg = "Attempt to exponentiate base of type Int64 to negative exponent";
              omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);  
              return new MsgTuple(errorMsg, MsgType.ERROR);                                
            }
            e.a= l.a**r.a;
          }     
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                              
            return new MsgTuple(errorMsg, MsgType.ERROR); 
          }
        }
      } else if e.etype == real {
        select op {
          // True division is the only integer type that would result in a
          // resultant type of `real`
          when "/" {
            e.a = l.a:real / r.a:real;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                              
            return new MsgTuple(errorMsg, MsgType.ERROR); 
          }   
        }
      }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    else if (e.etype == int && r.etype == uint) ||
            (e.etype == uint && r.etype == int) {
      select op {
        when ">>" {
          e.a = l.a >> r.a;
        }
        when "<<" {
          e.a = l.a << r.a;
        }
        when ">>>" {
          e.a = rotr(l.a, r.a);
        }
        when "<<<" {
          e.a = rotl(l.a, r.a);
        }
        otherwise {
          var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    } else if (l.etype == uint && r.etype == int) ||
              (l.etype == int && r.etype == uint) {
      select op {
        when "+" {
          e.a = l.a:real + r.a:real;
        }
        when "-" {
          e.a = l.a:real - r.a:real;
        }
        otherwise {
          var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                              
          return new MsgTuple(errorMsg, MsgType.ERROR); 
        }   
      }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    // If either RHS or LHS type is real, the same operations are supported and the
    // result will always be a `real`, so all 3 of these cases can be shared.
    else if ((l.etype == real && r.etype == real) || (l.etype == int && r.etype == real)
             || (l.etype == real && r.etype == int)) {
      select op {
          when "+" {
            e.a = l.a + r.a;
          }
          when "-" {
            e.a = l.a - r.a;
          }
          when "*" {
            e.a = l.a * r.a;
          }
          when "/" { // truediv
            e.a = l.a / r.a;
          } 
          when "//" { // floordiv
            ref ea = e.a;
            ref la = l.a;
            ref ra = r.a;
            [(ei,li,ri) in zip(ea,la,ra)] ei = floorDivisionHelper(li, ri);
          }
          when "**" { 
            e.a= l.a**r.a;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    } else if ((l.etype == uint && r.etype == real) || (l.etype == real && r.etype == uint)) {
      select op {
          when "+" {
            e.a = l.a:real + r.a:real;
          }
          when "-" {
            e.a = l.a:real - r.a:real;
          }
          when "*" {
            e.a = l.a:real * r.a:real;
          }
          when "/" { // truediv
            e.a = l.a:real / r.a:real;
          } 
          when "//" { // floordiv
            ref ea = e.a;
            ref la = l.a;
            ref ra = r.a;
            [(ei,li,ri) in zip(ea,la,ra)] ei = floorDivisionHelper(li, ri);
          }
          when "**" { 
            e.a= l.a:real**r.a:real;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    } else if ((l.etype == int && r.etype == bool) || (l.etype == bool && r.etype == int)) {
      select op {
          when "+" {
            // Since we don't know which of `l` or `r` is the int and which is the `bool`,
            // we can just cast both to int, which will be a noop for the vector that is
            // already `int`
            e.a = l.a:int + r.a:int;
          }
          when "-" {
            e.a = l.a:int - r.a:int;
          }
          when "*" {
            e.a = l.a:int * r.a:int;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    } else if ((l.etype == uint && r.etype == bool) || (l.etype == bool && r.etype == uint)) {
      select op {
          when "+" {
            e.a = l.a:uint + r.a:uint;
          }
          when "-" {
            e.a = l.a:uint - r.a:uint;
          }
          when "*" {
            e.a = l.a:uint * r.a:uint;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);  
    } else if ((l.etype == real && r.etype == bool) || (l.etype == bool && r.etype == real)) {
      select op {
          when "+" {
            e.a = l.a:real + r.a:real;
          }
          when "-" {
            e.a = l.a:real - r.a:real;
          }
          when "*" {
            e.a = l.a:real * r.a:real;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    var errorMsg = notImplementedError(pn,l.dtype,op,r.dtype);
    omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
    return new MsgTuple(errorMsg, MsgType.ERROR);
  }

  proc doBinOpvs(l, val, e, op: string, dtype, rname, pn, st) throws {
    if e.etype == bool {
      // Since we know that the result type is a boolean, we know
      // that it either (1) is an operation between bools or (2) uses
      // a boolean operator (<, <=, etc.)
      if l.etype == bool && val.type == bool {
        select op {
          when "|" {
            e.a = l.a | val;
          }
          when "&" {
            e.a = l.a & val;
          }
          when "^" {
            e.a = l.a ^ val;
          }
          when "==" {
            e.a = l.a == val;
          }
          when "!=" {
            e.a = l.a != val;
          }
          when "<" {
            e.a = l.a:int < val:int;
          }
          when ">" {
            e.a = l.a:int > val:int;
          }
          when "<=" {
            e.a = l.a:int <= val:int;
          }
          when ">=" {
            e.a = l.a:int >= val:int;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      }
      // All types support the same binary operations when the resultant
      // type is bool and `l` and `r` are not both boolean, so this does
      // not need to be specialized for each case.
      else {
        if ((l.etype == real && val.type == bool) || (l.etype == bool && val.type == real)) {
          select op {
            when "<" {
              e.a = l.a:real < val:real;
            }
            when ">" {
              e.a = l.a:real > val:real;
            }
            when "<=" {
              e.a = l.a:real <= val:real;
            }
            when ">=" {
              e.a = l.a:real >= val:real;
            }
            when "==" {
              e.a = l.a:real == val:real;
            }
            when "!=" {
              e.a = l.a:real != val:real;
            }
            otherwise {
              var errorMsg = notImplementedError(pn,l.dtype,op,dtype);
              omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
            }
          }
        }
        else {
          select op {
            when "<" {
              e.a = l.a < val;
            }
            when ">" {
              e.a = l.a > val;
            }
            when "<=" {
              e.a = l.a <= val;
            }
            when ">=" {
              e.a = l.a >= val;
            }
            when "==" {
              e.a = l.a == val;
            }
            when "!=" {
              e.a = l.a != val;
            }
            otherwise {
              var errorMsg = notImplementedError(pn,l.dtype,op,dtype);
              omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                              
              return new MsgTuple(errorMsg, MsgType.ERROR); 
            }
          }
        }
      }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    // Since we know that both `l` and `r` are of type `int` and that
    // the resultant type is not bool (checked in first `if`), we know
    // what operations are supported based on the resultant type
    else if (l.etype == int && val.type == int) ||
            (l.etype == uint && val.type == uint) {
      if e.etype == int || e.etype == uint {
        select op {
          when "+" {
            e.a = l.a + val;
          }
          when "-" {
            e.a = l.a - val;
          }
          when "*" {
            e.a = l.a * val;
          }
          when "//" { // floordiv
            ref ea = e.a;
            ref la = l.a;
            [(ei,li) in zip(ea,la)] ei = if val != 0 then li/val else 0;
          }
          when "%" { // modulo
            ref ea = e.a;
            ref la = l.a;
            [(ei,li) in zip(ea,la)] ei = if val != 0 then li%val else 0;
          }
          when "<<" {
            e.a = l.a << val;
          }                    
          when ">>" {
            e.a = l.a >> val;
          }
          when "<<<" {
            e.a = rotl(l.a, val);
          }
          when ">>>" {
            e.a = rotr(l.a, val);
          }
          when "&" {
            e.a = l.a & val;
          }                    
          when "|" {
            e.a = l.a | val;
          }                    
          when "^" {
            e.a = l.a ^ val;
          }
          when "**" { 
            e.a= l.a**val;
          }     
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                              
            return new MsgTuple(errorMsg, MsgType.ERROR); 
          }
        }
      } else if e.etype == real {
        select op {
          // True division is the only integer type that would result in a
          // resultant type of `real`
          when "/" {
            e.a = l.a:real / val:real;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                              
            return new MsgTuple(errorMsg, MsgType.ERROR); 
          }
            
        }
      }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    else if (e.etype == int && val.type == uint) ||
            (e.etype == uint && val.type == int) {
      select op {
        when ">>" {
          e.a = l.a >> val:l.etype;
        }
        when "<<" {
          e.a = l.a << val:l.etype;
        }
        when ">>>" {
          e.a = rotr(l.a, val:l.etype);
        }
        when "<<<" {
          e.a = rotl(l.a, val:l.etype);
        }
        when "+" {
          e.a = l.a + val:l.etype;
        }
        when "-" {
          e.a = l.a - val:l.etype;
        }
        otherwise {
          var errorMsg = notImplementedError(pn,l.dtype,op,dtype);
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    // If either RHS or LHS type is real, the same operations are supported and the
    // result will always be a `real`, so all 3 of these cases can be shared.
    else if ((l.etype == real && val.type == real) || (l.etype == int && val.type == real)
             || (l.etype == real && val.type == int)) {
      select op {
          when "+" {
            e.a = l.a + val;
          }
          when "-" {
            e.a = l.a - val;
          }
          when "*" {
            e.a = l.a * val;
          }
          when "/" { // truediv
            e.a = l.a / val;
          } 
          when "//" { // floordiv
            ref ea = e.a;
            ref la = l.a;
            [(ei,li) in zip(ea,la)] ei = floorDivisionHelper(li, val);
          }
          when "**" { 
            e.a= l.a**val;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    } else if ((l.etype == uint && val.type == real) || (l.etype == real && val.type == uint)) {
      select op {
          when "+" {
            e.a = l.a: real + val: real;
          }
          when "-" {
            e.a = l.a: real - val: real;
          }
          when "*" {
            e.a = l.a: real * val: real;
          }
          when "/" { // truediv
            e.a = l.a: real / val: real;
          } 
          when "//" { // floordiv
            ref ea = e.a;
            ref la = l.a;
            [(ei,li) in zip(ea,la)] ei = floorDivisionHelper(li, val);
          }
          when "**" { 
            e.a= l.a: real**val: real;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    } else if ((l.etype == int && val.type == bool) || (l.etype == bool && val.type == int)) {
      select op {
          when "+" {
            // Since we don't know which of `l` or `r` is the int and which is the `bool`,
            // we can just cast both to int, which will be a noop for the vector that is
            // already `int`
            e.a = l.a:int + val:int;
          }
          when "-" {
            e.a = l.a:int - val:int;
          }
          when "*" {
            e.a = l.a:int * val:int;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    } else if ((l.etype == real && val.type == bool) || (l.etype == bool && val.type == real)) {
      select op {
          when "+" {
            e.a = l.a:real + val:real;
          }
          when "-" {
            e.a = l.a:real - val:real;
          }
          when "*" {
            e.a = l.a:real * val:real;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,l.dtype,op,dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    var errorMsg = unrecognizedTypeError(pn, "("+dtype2str(l.dtype)+","+dtype2str(dtype)+")");
    omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
    return new MsgTuple(errorMsg, MsgType.ERROR);
  }

  proc doBinOpsv(val, r, e, op: string, dtype, rname, pn, st) throws {
    if e.etype == bool {
      // Since we know that the result type is a boolean, we know
      // that it either (1) is an operation between bools or (2) uses
      // a boolean operator (<, <=, etc.)
      if r.etype == bool && val.type == bool {
        select op {
          when "|" {
            e.a = val | r.a;
          }
          when "&" {
            e.a = val & r.a;
          }
          when "^" {
            e.a = val ^ r.a;
          }
          when "==" {
            e.a = val == r.a;
          }
          when "!=" {
            e.a = val != r.a;
          }
          when "<" {
            e.a = val:int < r.a:int;
          }
          when ">" {
            e.a = val:int > r.a:int;
          }
          when "<=" {
            e.a = val:int <= r.a:int;
          }
          when ">=" {
            e.a = val:int >= r.a:int;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      }
      // All types support the same binary operations when the resultant
      // type is bool and `l` and `r` are not both boolean, so this does
      // not need to be specialized for each case.
      else {
        if ((r.etype == real && val.type == bool) || (r.etype == bool && val.type == real)) {
          select op {
            when "<" {
              e.a = val:real < r.a:real;
            }
            when ">" {
              e.a = val:real > r.a:real;
            }
            when "<=" {
              e.a = val:real <= r.a:real;
            }
            when ">=" {
              e.a = val:real >= r.a:real;
            }
            when "==" {
              e.a = val:real == r.a:real;
            }
            when "!=" {
              e.a = val:real != r.a:real;
            }
            otherwise {
              var errorMsg = notImplementedError(pn,dtype,op,r.dtype);
              omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
            }
          }
        }
        else {
          select op {
            when "<" {
              e.a = val < r.a;
            }
            when ">" {
              e.a = val > r.a;
            }
            when "<=" {
              e.a = val <= r.a;
            }
            when ">=" {
              e.a = val >= r.a;
            }
            when "==" {
              e.a = val == r.a;
            }
            when "!=" {
              e.a = val != r.a;
            }
            otherwise {
              var errorMsg = notImplementedError(pn,dtype,op,r.dtype);
              omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                              
              return new MsgTuple(errorMsg, MsgType.ERROR); 
            }
          }
        }
      }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    // Since we know that both `l` and `r` are of type `int` and that
    // the resultant type is not bool (checked in first `if`), we know
    // what operations are supported based on the resultant type
    else if (r.etype == int && val.type == int) ||
            (r.etype == uint && val.type == uint) {
      if e.etype == int || e.etype == uint {
        select op {
          when "+" {
            e.a = val + r.a;
          }
          when "-" {
            e.a = val - r.a;
          }
          when "*" {
            e.a = val * r.a;
          }
          when "//" { // floordiv
            ref ea = e.a;
            ref ra = r.a;
            [(ei,ri) in zip(ea,ra)] ei = if ri != 0 then val/ri else 0;
          }
          when "%" { // modulo
            ref ea = e.a;
            ref ra = r.a;
            [(ei,ri) in zip(ea,ra)] ei = if ri != 0 then val%ri else 0;
          }
          when "<<" {
            e.a = val << r.a;
          }                    
          when ">>" {
            e.a = val >> r.a;
          }
          when "<<<" {
            e.a = rotl(val, r.a);
          }
          when ">>>" {
            e.a = rotr(val, r.a);
          }
          when "&" {
            e.a = val & r.a;
          }                    
          when "|" {
            e.a = val | r.a;
          }                    
          when "^" {
            e.a = val ^ r.a;
          }
          when "**" {
            if || reduce (r.a<0){
              var errorMsg = "Attempt to exponentiate base of type Int64 to negative exponent";
              omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR); 
            }
            e.a= val**r.a;
          }     
          otherwise {
            var errorMsg = notImplementedError(pn,dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                              
            return new MsgTuple(errorMsg, MsgType.ERROR); 
          }
        }
      } else if e.etype == real {
        select op {
          // True division is the only integer type that would result in a
          // resultant type of `real`
          when "/" {
            e.a = val:real / r.a:real;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                              
            return new MsgTuple(errorMsg, MsgType.ERROR); 
          }
            
        }
      }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    } else if (val.type == int && r.etype == uint) {
      select op {
        when ">>" {
          e.a = val:uint >> r.a:uint;
        }
        when "<<" {
          e.a = val:uint << r.a:uint;
        }
        when ">>>" {
          e.a = rotr(val:uint, r.a:uint);
        }
        when "<<<" {
          e.a = rotl(val:uint, r.a:uint);
        }
        when "+" {
          e.a = val:uint + r.a:uint;
        }
        when "-" {
          e.a = val:uint - r.a:uint;
        }
        otherwise {
          var errorMsg = notImplementedError(pn,dtype,op,r.dtype);
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    // If either RHS or LHS type is real, the same operations are supported and the
    // result will always be a `real`, so all 3 of these cases can be shared.
    else if ((r.etype == real && val.type == real) || (r.etype == int && val.type == real)
             || (r.etype == real && val.type == int)) {
      select op {
          when "+" {
            e.a = val + r.a;
          }
          when "-" {
            e.a = val - r.a;
          }
          when "*" {
            e.a = val * r.a;
          }
          when "/" { // truediv
            e.a = val:real / r.a:real;
          } 
          when "//" { // floordiv
            ref ea = e.a;
            ref ra = r.a;
            [(ei,ri) in zip(ea,ra)] ei = floorDivisionHelper(val:real, ri);
          }
          when "**" { 
            e.a= val**r.a;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    } else if ((r.etype == uint && val.type == real) || (r.etype == real && val.type == uint)) {
      select op {
          when "+" {
            e.a = val:real + r.a:real;
          }
          when "-" {
            e.a = val:real - r.a:real;
          }
          when "*" {
            e.a = val:real * r.a:real;
          }
          when "/" { // truediv
            e.a = val:real / r.a:real;
          } 
          when "//" { // floordiv
            ref ea = e.a;
            ref ra = r.a;
            [(ei,ri) in zip(ea,ra)] ei = floorDivisionHelper(val:real, ri);
          }
          when "**" { 
            e.a= val:real**r.a:real;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    } else if ((r.etype == int && val.type == bool) || (r.etype == bool && val.type == int)) {
      select op {
          when "+" {
            // Since we don't know which of `l` or `r` is the int and which is the `bool`,
            // we can just cast both to int, which will be a noop for the vector that is
            // already `int`
            e.a = val:int + r.a:int;
          }
          when "-" {
            e.a = val:int - r.a:int;
          }
          when "*" {
            e.a = val:int * r.a:int;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    } else if ((r.etype == real && val.type == bool) || (r.etype == bool && val.type == real)) {
      select op {
          when "+" {
            e.a = val:real + r.a:real;
          }
          when "-" {
            e.a = val:real - r.a:real;
          }
          when "*" {
            e.a = val:real * r.a:real;
          }
          otherwise {
            var errorMsg = notImplementedError(pn,dtype,op,r.dtype);
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    var errorMsg = unrecognizedTypeError(pn, "("+dtype2str(dtype)+","+dtype2str(r.dtype)+")");
    omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
    return new MsgTuple(errorMsg, MsgType.ERROR);
  }

  proc doBigIntBinOpvv(l, r, op: string) throws {
    var max_bits = max(l.max_bits, r.max_bits);
    var max_size = 1:bigint;
    var has_max_bits = max_bits != -1;
    if has_max_bits {
      max_size <<= max_bits;
    }
    ref la = l.a;
    ref ra = r.a;
    var tmp = makeDistArray(la.size, bigint);
    // these cases are not mutually exclusive,
    // so we have a flag to track if tmp is ever populated
    var visted = false;

    // had to create bigint specific BinOp procs which return
    // the distributed array because we need it at SymEntry creation time
    if l.etype == bigint && r.etype == bigint {
      // first we try the ops that only work with
      // both being bigint
      select op {
        when "&" {
          tmp = la & ra;
          visted = true;
        }
        when "|" {
          tmp = la | ra;
          visted = true;
        }
        when "^" {
          tmp = la ^ ra;
          visted = true;
        }
        when "/" {
          tmp = la / ra;
          visted = true;
        }
      }
    }
    if l.etype == bigint && (r.etype == bigint || r.etype == int || r.etype == uint) {
      // then we try the ops that only work with a
      // left hand side of bigint
      if r.etype != bigint {
        // can't shift a bigint by a bigint
        select op {
          when "<<" {
            tmp = la << ra;
            visted = true;
          }
          when ">>" {
            // workaround for right shift until chapel issue #21206
            // makes it into a release, eventually we can just do
            // tmp = la >> ra;
            var divideBy = makeDistArray(la.size, bigint);
            divideBy = 1:bigint;
            divideBy <<= ra;
            tmp = la / divideBy;
            visted = true;
          }
          when "<<<" {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotl");
            }
            // should be as simple as the below, see issue #2006
            // return (la << ra) | (la >> (max_bits - ra));
            tmp = la << ra;
            var botBits = la;
            if r.etype == int {
              var shift_amt = max_bits - ra;
              // cant just do botBits >>= shift_amt;
              var divideBy = makeDistArray(la.size, bigint);
              divideBy = 1:bigint;
              divideBy <<= shift_amt;
              botBits = botBits / divideBy;
              tmp += botBits;
            }
            else {
              var shift_amt = max_bits:uint - ra;
              botBits >>= shift_amt;
              tmp += botBits;
            }
            visted = true;
          }
          when ">>>" {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotr");
            }
            // should be as simple as the below, see issue #2006
            // return (la >> ra) | (la << (max_bits - ra));
            tmp = la;
            // cant just do tmp >>= ra;
            var divideBy = makeDistArray(la.size, bigint);
            divideBy = 1:bigint;
            divideBy <<= ra;
            tmp = tmp / divideBy;

            var topBits = la;
            if r.etype == int {
              var shift_amt = max_bits - ra;
              topBits <<= shift_amt;
              tmp += topBits;
            }
            else {
              var shift_amt = max_bits:uint - ra;
              topBits <<= shift_amt;
              tmp += topBits;
            }
            visted = true;
          }
        }
      }
      select op {
        when "//" { // floordiv
          [(ei,li,ri) in zip(tmp,la,ra)] ei = if ri != 0 then li/ri else 0:bigint;
          visted = true;
        }
        when "%" { // modulo
          // we only do in place mod when ri != 0, tmp will be 0 in other locations
          // we can't use ei = li % ri because this can result in negatives
          [(ei,li,ri) in zip(tmp,la,ra)] if ri != 0 then ei.mod(li, ri);
          visted = true;
        }
        when "**" {
          if || reduce (ra<0) {
            throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
          }
          if has_max_bits {
            [(ei,li,ri) in zip(tmp,la,ra)] ei.powMod(li, ri, max_size);
          }
          else {
            tmp = la ** ra;
          }
          visted = true;
        }
      }
    }
    if (l.etype == bigint && r.etype == bigint) ||
       (l.etype == bigint && (r.etype == int || r.etype == uint || r.etype == bool)) ||
       (r.etype == bigint && (l.etype == int || l.etype == uint || l.etype == bool)) {
      select op {
        when "+" {
          tmp = la + ra;
          visted = true;
        }
        when "-" {
          tmp = l.a - r.a;
          visted = true;
        }
        when "*" {
          tmp = l.a * r.a;
          visted = true;
        }
      }
    }
    if !visted {
      throw new Error("Unsupported operation: " + l.etype:string +" "+ op +" "+ r.etype:string);
    }
    else {
      if has_max_bits {
        // max_size should always be non-zero since we start at 1 and left shift
        tmp.mod(tmp, max_size);
      }
      return (tmp, max_bits);
    }
  }

  proc doBigIntBinOpvvBoolReturn(l, r, op: string) throws {
    select op {
      when "<" {
        return l.a < r.a;
      }
      when ">" {
        return l.a > r.a;
      }
      when "<=" {
        return l.a <= r.a;
      }
      when ">=" {
        return l.a >= r.a;
      }
      when "==" {
        return l.a == r.a;
      }
      when "!=" {
        return l.a != r.a;
      }
      otherwise {
        // we should never reach this since we only enter this proc
        // if boolOps.contains(op)
        throw new Error("Unsupported operation: " + l.etype:string +" "+ op +" "+ r.etype:string);
      }
    }
  }

  proc doBigIntBinOpvs(l, val, op: string) throws {
    var max_bits = l.max_bits;
    var max_size = 1:bigint;
    var has_max_bits = max_bits != -1;
    if has_max_bits {
      max_size <<= max_bits;
    }
    ref la = l.a;
    var tmp = makeDistArray(la.size, bigint);
    // these cases are not mutually exclusive,
    // so we have a flag to track if tmp is ever populated
    var visted = false;

    // had to create bigint specific BinOp procs which return
    // the distributed array because we need it at SymEntry creation time
    if l.etype == bigint && val.type == bigint {
      // first we try the ops that only work with
      // both being bigint
      select op {
        when "&" {
          tmp = la & val;
          visted = true;
        }
        when "|" {
          tmp = la | val;
          visted = true;
        }
        when "^" {
          tmp = la ^ val;
          visted = true;
        }
        when "/" {
          tmp = la / val;
          visted = true;
        }
      }
    }
    if l.etype == bigint && (val.type == bigint || val.type == int || val.type == uint) {
      // then we try the ops that only work with a
      // left hand side of bigint
      if val.type != bigint {
        // can't shift a bigint by a bigint
        select op {
          when "<<" {
            tmp = la << val;
            visted = true;
          }
          when ">>" {
            // workaround for right shift until chapel issue #21206
            // makes it into a release, eventually we can just do
            // tmp = la >> ra;
            var divideBy = makeDistArray(la.size, bigint);
            divideBy = 1:bigint;
            divideBy <<= val;
            tmp = la / divideBy;
            visted = true;
          }
          when "<<<" {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotl");
            }
            // should be as simple as the below, see issue #2006
            // return (la << val) | (la >> (max_bits - val));
            tmp = la << val;
            var botBits = la;
            if val.type == int {
              var shift_amt = max_bits - val;
              // cant just do botBits >>= shift_amt;
              var divideBy = makeDistArray(la.size, bigint);
              divideBy = 1:bigint;
              divideBy <<= shift_amt;
              botBits = botBits / divideBy;
              tmp += botBits;
            }
            else {
              var shift_amt = max_bits:uint - val;
              botBits >>= shift_amt;
              tmp += botBits;
            }
            visted = true;
          }
          when ">>>" {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotr");
            }
            // should be as simple as the below, see issue #2006
            // return (la >> val) | (la << (max_bits - val));
            tmp = la;
            // cant just do tmp >>= ra;
            var divideBy = makeDistArray(la.size, bigint);
            divideBy = 1:bigint;
            divideBy <<= val;
            tmp = tmp / divideBy;

            var topBits = la;
            if val.type == int {
              var shift_amt = max_bits - val;
              topBits <<= shift_amt;
              tmp += topBits;
            }
            else {
              var shift_amt = max_bits:uint - val;
              topBits <<= shift_amt;
              tmp += topBits;
            }
            visted = true;
          }
        }
      }
      select op {
        when "//" { // floordiv
          [(ei,li) in zip(tmp,la)] ei = if val != 0 then li/val else 0:bigint;
          visted = true;
        }
        when "%" { // modulo
          // we only do in place mod when val != 0, tmp will be 0 in other locations
          // we can't use ei = li % val because this can result in negatives
          [(ei,li) in zip(tmp,la)] if val != 0 then ei.mod(li, val);
          visted = true;
        }
        when "**" {
          if val<0 {
            throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
          }
          if has_max_bits {
            [(ei,li) in zip(tmp,la)] ei.powMod(li, val, max_size);
          }
          else {
            tmp = la ** val;
          }
          visted = true;
        }
      }
    }
    if (l.etype == bigint && val.type == bigint) ||
       (l.etype == bigint && (val.type == int || val.type == uint || val.type == bool)) ||
       (val.type == bigint && (l.etype == int || l.etype == uint || l.etype == bool)) {
      select op {
        when "+" {
          tmp = la + val;
          visted = true;
        }
        when "-" {
          tmp = l.a - val;
          visted = true;
        }
        when "*" {
          tmp = l.a * val;
          visted = true;
        }
      }
    }
    if !visted {
      throw new Error("Unsupported operation: " + l.etype:string +" "+ op +" "+ val.type:string);
    }
    else {
      if has_max_bits {
        // max_size should always be non-zero since we start at 1 and left shift
        tmp.mod(tmp, max_size);
      }
      return (tmp, max_bits);
    }
  }

  proc doBigIntBinOpvsBoolReturn(l, val, op: string) throws {
    select op {
      when "<" {
        return l.a < val;
      }
      when ">" {
        return l.a > val;
      }
      when "<=" {
        return l.a <= val;
      }
      when ">=" {
        return l.a >= val;
      }
      when "==" {
        return l.a == val;
      }
      when "!=" {
        return l.a != val;
      }
      otherwise {
        // we should never reach this since we only enter this proc
        // if boolOps.contains(op)
        throw new Error("Unsupported operation: " +" "+ l.etype:string + op +" "+ val.type:string);
      }
    }
  }

  proc doBigIntBinOpsv(val, r, op: string) throws {
    var max_bits = r.max_bits;
    var max_size = 1:bigint;
    var has_max_bits = max_bits != -1;
    if has_max_bits {
      max_size <<= max_bits;
    }
    ref ra = r.a;
    var tmp = makeDistArray(ra.size, bigint);
    // these cases are not mutually exclusive,
    // so we have a flag to track if tmp is ever populated
    var visted = false;

    // had to create bigint specific BinOp procs which return
    // the distributed array because we need it at SymEntry creation time
    if val.type == bigint && r.etype == bigint {
      // first we try the ops that only work with
      // both being bigint
      select op {
        when "&" {
          tmp = val & ra;
          visted = true;
        }
        when "|" {
          tmp = val | ra;
          visted = true;
        }
        when "^" {
          tmp = val ^ ra;
          visted = true;
        }
        when "/" {
          tmp = val / ra;
          visted = true;
        }
      }
    }
    if val.type == bigint && (r.etype == bigint || r.etype == int || r.etype == uint) {
      // then we try the ops that only work with a
      // left hand side of bigint
      if r.etype != bigint {
        // can't shift a bigint by a bigint
        select op {
          when "<<" {
            tmp = val << ra;
            visted = true;
          }
          when ">>" {
            // workaround for right shift until chapel issue #21206
            // makes it into a release, eventually we can just do
            // tmp = val >> ra;
            var divideBy = makeDistArray(ra.size, bigint);
            divideBy = 1:bigint;
            divideBy <<= ra;
            tmp = val / divideBy;
            visted = true;
          }
          when "<<<" {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotl");
            }
            // should be as simple as the below, see issue #2006
            // return (val << ra) | (val >> (max_bits - ra));
            tmp = val << ra;
            var botBits = makeDistArray(ra.size, bigint);
            botBits = val;
            if r.etype == int {
              var shift_amt = max_bits - ra;
              // cant just do botBits >>= shift_amt;
              var divideBy = makeDistArray(ra.size, bigint);
              divideBy = 1:bigint;
              divideBy <<= shift_amt;
              botBits = botBits / divideBy;
              tmp += botBits;
            }
            else {
              var shift_amt = max_bits:uint - ra;
              botBits >>= shift_amt;
              tmp += botBits;
            }
            visted = true;
          }
          when ">>>" {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotr");
            }
            // should be as simple as the below, see issue #2006
            // return (val >> ra) | (val << (max_bits - ra));
            tmp = val;
            // cant just do tmp >>= ra;
            var divideBy = makeDistArray(ra.size, bigint);
            divideBy = 1:bigint;
            divideBy <<= ra;
            tmp = tmp / divideBy;

            var topBits = makeDistArray(ra.size, bigint);
            topBits = val;
            if r.etype == int {
              var shift_amt = max_bits - ra;
              topBits <<= shift_amt;
              tmp += topBits;
            }
            else {
              var shift_amt = max_bits:uint - ra;
              topBits <<= shift_amt;
              tmp += topBits;
            }
            visted = true;
          }
        }
      }
      select op {
        when "//" { // floordiv
          [(ei,ri) in zip(tmp,ra)] ei = if ri != 0 then val/ri else 0:bigint;
          visted = true;
        }
        when "%" { // modulo
          // we only do in place mod when val != 0, tmp will be 0 in other locations
          // we can't use ei = li % val because this can result in negatives
          [(ei,ri) in zip(tmp,ra)] if ri != 0 then ei.mod(val, ri);
          visted = true;
        }
        when "**" {
          if || reduce(ra<0) {
            throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
          }
          if has_max_bits {
            [(ei,ri) in zip(tmp,ra)] ei.powMod(val, ri, max_size);
          }
          else {
            tmp = val ** ra;
          }
          visted = true;
        }
      }
    }
    if (val.type == bigint && r.etype == bigint) ||
       (val.type == bigint && (r.etype == int || r.etype == uint || r.etype == bool)) ||
       (r.etype == bigint && (val.type == int || val.type == uint || val.type == bool)) {
      // TODO we have to cast to bigint until chape issue #21290 is resolved, see issue #2007
      var cast_val = if val.type == bool then val:int:bigint else val:bigint;
      select op {
        when "+" {
          tmp = cast_val + ra;
          visted = true;
        }
        when "-" {
          tmp = cast_val - r.a;
          visted = true;
        }
        when "*" {
          tmp = cast_val * r.a;
          visted = true;
        }
      }
    }
    if !visted {
      throw new Error("Unsupported operation: " + val.type:string +" "+ op +" "+ r.etype:string);
    }
    else {
      if has_max_bits {
        // max_size should always be non-zero since we start at 1 and left shift
        tmp.mod(tmp, max_size);
      }
      return (tmp, max_bits);
    }
  }

  proc doBigIntBinOpsvBoolReturn(val, r, op: string) throws {
    select op {
      when "<" {
        return val < r.a;
      }
      when ">" {
        return val > r.a;
      }
      when "<=" {
        return val <= r.a;
      }
      when ">=" {
        return val >= r.a;
      }
      when "==" {
        return val == r.a;
      }
      when "!=" {
        return val != r.a;
      }
      otherwise {
        // we should never reach this since we only enter this proc
        // if boolOps.contains(op)
        throw new Error("Unsupported operation: " + val.type:string +" "+ op +" "+ r.etype:string);
      }
    }
  }
}
