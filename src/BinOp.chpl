
module BinOp
{
  use ServerConfig;
  
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Logging;
  use Message;

  private config const logLevel = ServerConfig.logLevel;
  const omLogger = new Logger(logLevel);

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
    // Since we know that the result type is a boolean, we know
    // that it either (1) is an operation between bools or (2) uses
    // one of the boolean operators from the `boolOps` set above
    if e.etype == bool {
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
      var repMsg = "created %s".format(st.attrib(rname));
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    // Since we know that both `l` and `r` are of type `int` and that
    // the resultant type is not bool (checked in first `if`), we know
    // what operations are supported based on the resultant type
    else if l.etype == int && r.etype == int {
      if e.etype == int {
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
            [(ei,li,ri) in zip(ea,la,ra)] ei = if ri != 0 then floor(li/ri) else NAN;
          }
          when "**" { 
            e.a= l.a**r.a;
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
    return new MsgTuple("Bin op not supported", MsgType.NORMAL);
  }
}
