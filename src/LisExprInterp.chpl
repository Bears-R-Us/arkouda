module LisExprInterp
{

    use LisExprData;

    /*
      tokenize the prog
    */
    proc tokenize(line: string) {
        var l: list(string) = line.replace("("," ( ").replace(")"," ) ").split();
        return l;
    }
    
    /*
      parse, check, and validate code and all symbols in the tokenized prog
    */ 
    proc parse(line: string): GenListValue throws {
        // Want:
        //   return read_from_tokens(tokenize(line));
        //
        // Workaround (see https://github.com/chapel-lang/chapel/issues/16170):

        var l: list(string) = tokenize(line);
        return read_from_tokens(l);
    }
    
    /*
      parse throught the list of tokens generating the parse tree / AST
      as a list of atoms and lists
    */
    proc read_from_tokens(ref tokens: list(string)): GenListValue throws {
        if (tokens.size == 0) then
            throw new owned Error("SyntaxError: unexpected EOF");

        // Open Q: If we were to parse from the back of the string to the
        // front, could this be more efficient since popping from the
        // front of a list is an expensive operation?

        var token = tokens.pop(0);
        if (token == "(") {
            var L = new GenList();
            while (tokens.first() != ")") {
                L.append(read_from_tokens(tokens));
                if (tokens.size == 0) then
                    throw new owned Error("SyntaxError: unexpected EOF");
            }
            tokens.pop(0); // pop off ")"
            return new ListValue(L);
        }
        else if (token == ")") {
            throw new owned Error("SyntaxError: unexpected )");
        }
        else {
            return atom(token);
        }
    }
    
    /* determine atom type and values */
    proc atom(token: string): GenListValue {
        try { // try to interpret as an integer ?
            return new ListValue(token:int); 
        } catch {
            try { //try to interpret it as a real ?
                return new ListValue(token:real);
            } catch { // return it as a symbol
                return new ListValue(token);
            }
        }
    }
    
    /* check to see if list value is a symbol otherwise throw error */
    inline proc checkSymbol(arg: BGenListValue) throws {
        if (arg.lvt != LVT.Sym) {
          throw new owned Error("arg must be a symbol " + arg:string);
        }
    }

    /* check to see if size is greater than or equal to size otherwise throw error */
    inline proc checkGEqLstSize(lst: GenList, sz: int) throws {
        if (lst.size < sz) {
          throw new owned Error("list must be at least size " + sz:string + " " + lst:string);
        }
    }

    /* check to see if size is equal to size otherwise throw error */
    inline proc checkEqLstSize(lst: GenList, sz: int) throws {
        if (lst.size != sz) {
          throw new owned Error("list must be size" + sz:string + " " +  lst:string);
        }
    }

    /*
      evaluate the expression
    */
    proc eval(ast: BGenListValue, env: borrowed Env, st): GenValue throws {
        select (ast.lvt) {
            when (LVT.Sym) {
                var gv = env.lookup(ast.toListValue(Symbol).lv);
                return gv.copy();
            }
            when (LVT.I) {
                var ret: int = ast.toListValue(int).lv;
                return new Value(ret);
            }
            when (LVT.R) {
                var ret: real = ast.toListValue(real).lv;
                return new Value(ret);
            }
            when (LVT.Lst) {
                ref lst = ast.toListValue(GenList).lv;
                // no empty lists allowed
                checkGEqLstSize(lst,1);
                // currently first list element must be a symbol of operator
                checkSymbol(lst[0]);
                var op = lst[0].toListValue(Symbol).lv;
                select (op) {
                    when "+"   {checkEqLstSize(lst,3); return eval(lst[1], env, st) + eval(lst[2], env, st);}
                    when "-"   {checkEqLstSize(lst,3); return eval(lst[1], env, st) - eval(lst[2], env, st);}
                    when "*"   {checkEqLstSize(lst,3); return eval(lst[1], env, st) * eval(lst[2], env, st);}
                    when "=="  {checkEqLstSize(lst,3); return eval(lst[1], env, st) == eval(lst[2], env, st);}
                    when "!="  {checkEqLstSize(lst,3); return eval(lst[1], env, st) != eval(lst[2], env, st);}
                    when "<"   {checkEqLstSize(lst,3); return eval(lst[1], env, st) < eval(lst[2], env, st);}
                    when "<="  {checkEqLstSize(lst,3); return eval(lst[1], env, st) <= eval(lst[2], env, st);}
                    when ">"   {checkEqLstSize(lst,3); return eval(lst[1], env, st) > eval(lst[2], env, st);}
                    when ">="  {checkEqLstSize(lst,3); return eval(lst[1], env, st) >= eval(lst[2], env, st);}
                    when "or"  {checkEqLstSize(lst,3); return or(eval(lst[1], env, st), eval(lst[2], env, st));}
                    when "and" {checkEqLstSize(lst,3); return and(eval(lst[1], env, st), eval(lst[2], env, st));}
                    when "not" {checkEqLstSize(lst,2); return not(eval(lst[1], env, st));}
                    when ":=" {
                        checkEqLstSize(lst,3);
                        checkSymbol(lst[1]);
                        var name = lst[1].toListValue(Symbol).lv;
                        // addEnrtry redefines values for already existing entries
                        var gv = env.addEntry(name, eval(lst[2],env, st));
                        //TODO: how to continue evaling after an assignment?
                        return gv.copy(); // return value assigned to symbol
                    }
                    when "lookup_and_index_float64" {
                        var entry = st.lookup(lst[1].toListValue(Symbol).lv);
                        var e = toSymEntry(toGenSymEntry(entry), real);
                        var i = eval(lst[2],env,st).toValue(int).v;
                        return new Value(e.a[i]);
                    }
                    when "lookup_and_index_int64" {
                        var entry = st.lookup(lst[1].toListValue(Symbol).lv);
                        var e = toSymEntry(toGenSymEntry(entry), int);
                        var i = eval(lst[2],env,st).toValue(int).v;
                        return new Value(e.a[i]);
                    }
                    when "if" {
                        checkEqLstSize(lst,4);
                        if isTrue(eval(lst[1], env, st)) {return eval(lst[2], env, st);} else {return eval(lst[3], env, st);}
                    }
                    when "begin" {
                      checkGEqLstSize(lst, 1);
                      // setup the environment
                      for i in 1..#lst.size-1 do
                        eval(lst[i], env, st);
                      // eval the return expression
                      return eval(lst[lst.size-1], env, st);
                    }
                    when "return" { // for now, just eval the next line, in time, might want to coerce return value
                        return eval(lst[1], env, st);
                    }
                    otherwise {
                        throw new owned Error("op not implemented " + op);
                    }
                }
            }
            otherwise {
              throw new owned Error("undefined ast node type " + ast:string);
            }
        }
    }


}
