module LisExpr
{
    class ErrorWithMsg: Error
    {
        var msg: string;
    }
    
    public use List;
    
    /* list value type */
    enum LVT {Lst, Sym, I, R};
    
    type Symbol = string;

    /* type: list of genric list values */
    type GenList = list(owned GenListValue);

    /* generic list value */
    class GenListValue
    {
        var lvt: LVT;
        
        /* initialize the list value type so we can test it at runtime */
        proc init(type lvtype) {
            if (lvtype == list(GenListValue)) {lvt = LVT.Lst;}
            if (lvtype == Symbol)             {lvt = LVT.Sym;}
            if (lvtype == int)                {lvt = LVT.I;}
            if (lvtype == real)               {lvt = LVT.R;}
        }
        
        /* cast to the GenListValue to borrowed ListValue(vtype) halt on failure */
        inline proc toListValue(type lvtype) {
            return try! this :borrowed ListValue(lvtype);
        }

        proc copy(): owned GenListValue throws {
          select (this.lvt) {
            when (LVT.Lst) {
              //              var copyList = copyOwnedList(this.toListValue(list(owned GenListValue)).lv);
              var copyList = copyOwnedList(this.toListValue(GenList).lv);
              return new owned ListValue(copyList);
            }
            when (LVT.Sym) {
              return new owned ListValue(this.toListValue(Symbol).lv);
            }
            when (LVT.I) {
              return new owned ListValue(this.toListValue(int).lv);
            }
            when (LVT.R) {
              return new owned ListValue(this.toListValue(real).lv);
            }
            otherwise {throw new owned ErrorWithMsg("not implemented");}
          }
        }
        
    }
    
    /* concrete list value */
    class ListValue : GenListValue
    {
        type lvtype;
        var lv: lvtype;
        
        /* initialize the value and the vtype */
        proc init(val: ?vtype) {
            super.init(vtype);
            this.lvtype = vtype;
            this.complete();
            if (isListType(vtype)) {
              try! copyOwnedList(this.lv, val);
            } else {
              this.lv = val;
            }
        }
        
    }

    proc isListType(type t: list(?)) param {
      return true;
    }

    proc isListType(type t) param {
      return false;
    }

    proc copyOwnedList(ref dst: list(?t,?p), src: list(t, ?p2)) throws {
      for item in src {
        dst.append(item.copy());
      }
    }

    proc copyOwnedList(src: list(?t, ?p)): list(t, p) throws {
      var dst: list(t, p);
      for item in src {
        dst.append(item.copy());
      }
      return dst;
    }

    
    /*
    proc (list(owned GenListValue, ?)).init=(lhs: list(owned GenListValue, ?p)) {
      compilerWarning("In my initializer");
      for l in lhs do
        this.append(l.copy());
    }
    */
        
    
    // allowed value types int and real
    enum VT {I, R};

    /* generic value class */
    class GenValue
    {
        /* value type testable at runtime */
        var vt: VT;
    
        /* initialize the value type so we can test it at runtime */
        proc init(type vtype) {
            if (vtype == int)  {vt = VT.I;}
            if (vtype == real) {vt = VT.R;}
        }
        
        /* cast to the GenValue to borrowed Value(vtype) halt on failure */
        inline proc toValue(type vtype) {
            return try! this :borrowed Value(vtype);
        }

        /* returns a copy of this... an owned GenValue */
        proc copy(): owned GenValue throws {
            select (this.vt) {
                when (VT.I) {return new owned Value(this.toValue(int).v);}
                when (VT.R) {return new owned Value(this.toValue(real).v);}
                otherwise { throw new owned ErrorWithMsg("not implemented"); }
            }
        }
    }
    
    /* concrete value class */
    class Value : GenValue
    {
        type vtype; // value type
        var v: vtype; // value
        
        /* initialize the value and the vtype */
        proc init(val: ?vtype) {
            super.init(vtype);
            this.vtype = vtype;
            this.v = val;
        }
    }

    //////////////////////////////////////////
    // operators over GenValue
    //////////////////////////////////////////
    
    inline proc +(l: borrowed GenValue, r: borrowed GenValue): owned GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new owned Value(l.toValue(int).v + r.toValue(int).v);}
            when (VT.I, VT.R) {return new owned Value(l.toValue(int).v + r.toValue(real).v);}
            when (VT.R, VT.I) {return new owned Value(l.toValue(real).v + r.toValue(int).v);}
            when (VT.R, VT.R) {return new owned Value(l.toValue(real).v + r.toValue(real).v);}
            otherwise {throw new owned ErrorWithMsg("not implemented");}
        }
    }

    inline proc -(l: borrowed GenValue, r: borrowed GenValue): owned GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new owned Value(l.toValue(int).v - r.toValue(int).v);}
            when (VT.I, VT.R) {return new owned Value(l.toValue(int).v - r.toValue(real).v);}
            when (VT.R, VT.I) {return new owned Value(l.toValue(real).v - r.toValue(int).v);}
            when (VT.R, VT.R) {return new owned Value(l.toValue(real).v - r.toValue(real).v);}
            otherwise {throw new owned ErrorWithMsg("not implemented");}
        }
    }

    inline proc *(l: borrowed GenValue, r: borrowed GenValue): owned GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new owned Value(l.toValue(int).v * r.toValue(int).v);}
            when (VT.I, VT.R) {return new owned Value(l.toValue(int).v * r.toValue(real).v);}
            when (VT.R, VT.I) {return new owned Value(l.toValue(real).v * r.toValue(int).v);}
            when (VT.R, VT.R) {return new owned Value(l.toValue(real).v * r.toValue(real).v);}
            otherwise {throw new owned ErrorWithMsg("not implemented");}
        }
    }

    inline proc <(l: borrowed GenValue, r: borrowed GenValue): owned GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new owned Value((l.toValue(int).v < r.toValue(int).v):int);}
            when (VT.I, VT.R) {return new owned Value((l.toValue(int).v < r.toValue(real).v):int);}
            when (VT.R, VT.I) {return new owned Value((l.toValue(real).v < r.toValue(int).v):int);}
            when (VT.R, VT.R) {return new owned Value((l.toValue(real).v < r.toValue(real).v):int);}
            otherwise {throw new owned ErrorWithMsg("not implemented");}
        }
    }

    inline proc >(l: borrowed GenValue, r: borrowed GenValue): owned GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new owned Value((l.toValue(int).v > r.toValue(int).v):int);}
            when (VT.I, VT.R) {return new owned Value((l.toValue(int).v > r.toValue(real).v):int);}
            when (VT.R, VT.I) {return new owned Value((l.toValue(real).v > r.toValue(int).v):int);}
            when (VT.R, VT.R) {return new owned Value((l.toValue(real).v > r.toValue(real).v):int);}
            otherwise {throw new owned ErrorWithMsg("not implemented");}
        }
    }

    inline proc <=(l: borrowed GenValue, r: borrowed GenValue): owned GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new owned Value((l.toValue(int).v <= r.toValue(int).v):int);}
            when (VT.I, VT.R) {return new owned Value((l.toValue(int).v <= r.toValue(real).v):int);}
            when (VT.R, VT.I) {return new owned Value((l.toValue(real).v <= r.toValue(int).v):int);}
            when (VT.R, VT.R) {return new owned Value((l.toValue(real).v <= r.toValue(real).v):int);}
            otherwise {throw new owned ErrorWithMsg("not implemented");}
        }
    }

    inline proc >=(l: borrowed GenValue, r: borrowed GenValue): owned GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new owned Value((l.toValue(int).v >= r.toValue(int).v):int);}
            when (VT.I, VT.R) {return new owned Value((l.toValue(int).v >= r.toValue(real).v):int);}
            when (VT.R, VT.I) {return new owned Value((l.toValue(real).v >= r.toValue(int).v):int);}
            when (VT.R, VT.R) {return new owned Value((l.toValue(real).v >= r.toValue(real).v):int);}
            otherwise {throw new owned ErrorWithMsg("not implemented");}
        }
    }

    inline proc ==(l: borrowed GenValue, r: borrowed GenValue): owned GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new owned Value((l.toValue(int).v == r.toValue(int).v):int);}
            when (VT.I, VT.R) {return new owned Value((l.toValue(int).v == r.toValue(real).v):int);}
            when (VT.R, VT.I) {return new owned Value((l.toValue(real).v == r.toValue(int).v):int);}
            when (VT.R, VT.R) {return new owned Value((l.toValue(real).v == r.toValue(real).v):int);}
            otherwise {throw new owned ErrorWithMsg("not implemented");}
        }
    }

    inline proc !=(l: borrowed GenValue, r: borrowed GenValue): owned GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new owned Value((l.toValue(int).v != r.toValue(int).v):int);}
            when (VT.I, VT.R) {return new owned Value((l.toValue(int).v != r.toValue(real).v):int);}
            when (VT.R, VT.I) {return new owned Value((l.toValue(real).v != r.toValue(int).v):int);}
            when (VT.R, VT.R) {return new owned Value((l.toValue(real).v != r.toValue(real).v):int);}
            otherwise {throw new owned ErrorWithMsg("not implemented");}
        }
    }

    inline proc and(l: borrowed GenValue, r: borrowed GenValue): owned GenValue throws {
        return new owned Value((l && r):int);
    }

    inline proc or(l: borrowed GenValue, r: borrowed GenValue): owned GenValue throws {
        return new owned Value((l || r):int);
    }

    inline proc not(l: borrowed GenValue): owned GenValue throws {
        return new owned Value((! isTrue(l)):int);
    }

    inline proc isTrue(gv: borrowed GenValue): bool throws {
        select (gv.vt) {
            when (VT.I) {return (gv.toValue(int).v != 0);}
            when (VT.R) {return (gv.toValue(real).v != 0.0);}
            otherwise {throw new owned ErrorWithMsg("not implemented");}
        }
    }
    
    /* environment is a dictionary of {string:GenValue} */
    class Env
    {
        /* what data structure to use ??? assoc array over strings or a map ? */
        var tD: domain(string);
        var tab: [tD] owned GenValue?;

        /* add a new entry or set an entry to a new value */
        proc addEntry(name:string, val: ?t): borrowed Value(t) throws {
            var entry = new owned Value(val);
            if (!tD.contains(name)) {tD += name;};
            ref tableEntry = tab[name];
            tableEntry = entry;
            return tableEntry!.borrow().toValue(t);
        }

        /* add a new entry or set an entry to a new value */
        proc addEntry(name:string, in entry: owned GenValue): borrowed GenValue throws {
            if (!tD.contains(name)) {tD += name;};
            ref tableEntry = tab[name];
            tableEntry = entry;
            return tableEntry!.borrow();
        }

        /* lookup symbol and throw error if not found */
        proc lookup(name: string): borrowed GenValue throws {
            if (!tD.contains(name) || tab[name] == nil) {
                throw new owned ErrorWithMsg("undefined symbol error (%t)".format(name));
            }
            return tab[name]!;
        }

        /* delete entry -- not sure if we need this */
        proc deleteEntry(name: string) {
            use IO;
            if (tD.contains(name)) {
                tab[name] = nil;
                tD -= name;
            }
            else {
                writeln("unkown symbol ",name);
                try! stdout.flush();
            }
        }
    }

    //////////////////////////////////////////////////////////////////
    // above this is stuff to support generic lists and generic values
    // and environment(dictionary/map) of symbols to values
    //////////////////////////////////////////////////////////////////
    
    /*
      tokenize the prog
    */
    proc tokenize(line: string) {
      var l: list(string);
      for token in line.replace("("," ( ").replace(")"," ) ").split() do
        l.append(token);
      return l;
    }
    
    /*
      parse, check, and validate code and all symbols in the tokenized prog
    */ 
    proc parse(line: string): owned GenListValue throws {
                                                         var l: list(string) = tokenize(line);
                                                         
        return read_from_tokens(l);
    }
    
    /*
      parse throught the list of tokens generating the parse tree / AST
      as a list of atoms and lists
    */
    proc read_from_tokens(ref tokens: list(string)): owned GenListValue throws {
                                                                                //writeln("In read_from_tokens(", tokens, ")");
        if (tokens.size == 0) then
            throw new owned ErrorWithMsg("SyntaxError: unexpected EOF");
        //                                                                                                                                                                    writeln("pre-pop: ", tokens);
                                                                                                                                                                    
        var token = tokens.pop(0);
        //                                                                                                                                                                    writeln("post-pop: ", tokens);
                                                                                                                                                                    
        if (token == "(") {
          //          writeln("inside conditional: ", tokens);
          
            var L: GenList;
            //            writeln("before while loop: ", tokens);
            
            while (tokens.first() != ")") {
              //              writeln("inside while loop: ", tokens);
              
              //              writeln("calling recursively with: ", tokens);
              
                L.append(read_from_tokens(tokens));
                //                writeln("returned from read_from_tokens()");
                
                if (tokens.size == 0) then
                    throw new owned ErrorWithMsg("SyntaxError: unexpected EOF");
            }
            tokens.pop(0); // pop off ")"
            return new owned ListValue(L);
        }
        else if (token == ")") {
            throw new owned ErrorWithMsg("SyntaxError: unexpected )");
        }
        else {
            return atom(token);
        }
    }
    
    /* determine atom type and values */
    proc atom(token: string): owned GenListValue {
        try { // try to interpret as an integer ?
            return new owned ListValue(token:int); 
        } catch {
            try { //try to interpret it as a real ?
                return new owned ListValue(token:real);
            } catch { // return it as a symbol
                return new owned ListValue(token);
            }
        }
    }
    
    /* check to see if list value is a symbol otherwise throw error */
    inline proc checkSymbol(arg: borrowed GenListValue) throws {
        if (arg.lvt != LVT.Sym) {
            throw new owned ErrorWithMsg("arg must be a symbol %t".format(arg));
        }
    }

    /* check to see if size is greater than or equal to size otherwise throw error */
    inline proc checkGEqLstSize(lst: GenList, sz: int) throws {
        if (lst.size < sz) {
            throw new owned ErrorWithMsg("list must be at least size %t %t".format(sz, lst));
        }
    }

    /* check to see if size is equal to size otherwise throw error */
    inline proc checkEqLstSize(lst: GenList, sz: int) throws {
        if (lst.size != sz) {
            throw new owned ErrorWithMsg("list must be size %t %t".format(sz, lst));
        }
    }

    /*
      evaluate the expression
    */
    proc eval(ast: borrowed GenListValue, env: borrowed Env): owned GenValue throws {
        select (ast.lvt) {
            when (LVT.Sym) {
                var gv = env.lookup(ast.toListValue(Symbol).lv);
                return gv.copy();
            }
            when (LVT.I) {
                var ret: int = ast.toListValue(int).lv;
                return new owned Value(ret);
            }
            when (LVT.R) {
                var ret: real = ast.toListValue(real).lv;
                return new owned Value(ret);
            }
            when (LVT.Lst) {
                //                ref lst = ast.toListValue(GenList).lv;
                var lst = copyOwnedList(ast.toListValue(GenList).lv);            
                // no empty lists allowed
                checkGEqLstSize(lst,1);
                // currently first list element must be a symbol of operator
                checkSymbol(lst[0]);
                var op = lst[0].toListValue(Symbol).lv;
                select (op) {
                    when "+"  {checkEqLstSize(lst,3); return eval(lst[1], env) + eval(lst[2], env);}
                    when "-"  {checkEqLstSize(lst,3); return eval(lst[1], env) - eval(lst[2], env);}
                    when "*"  {checkEqLstSize(lst,3); return eval(lst[1], env) * eval(lst[2], env);}
                    when "<"  {checkEqLstSize(lst,3); return eval(lst[1], env) < eval(lst[2], env);}
                    when ">"  {checkEqLstSize(lst,3); return eval(lst[1], env) > eval(lst[2], env);}
                    when "<="  {checkEqLstSize(lst,3); return eval(lst[1], env) <= eval(lst[2], env);}
                    when ">="  {checkEqLstSize(lst,3); return eval(lst[1], env) >= eval(lst[2], env);}
                    when "==" {checkEqLstSize(lst,3); return eval(lst[1], env) == eval(lst[2], env);}
                    when "!=" {checkEqLstSize(lst,3); return eval(lst[1], env) != eval(lst[2], env);}
                    when "or" {checkEqLstSize(lst,3); return or(eval(lst[1], env), eval(lst[2], env));}
                    when "and" {checkEqLstSize(lst,3); return and(eval(lst[1], env), eval(lst[2], env));}
                    when "not" {checkEqLstSize(lst,2); return not(eval(lst[1], env));}
                    when "set!" {
                        checkEqLstSize(lst,3);
                        checkSymbol(lst[1]);
                        var name = lst[1].toListValue(Symbol).lv;
                        // addEnrtry redefines values for already existing entries
                        var gv = env.addEntry(name, eval(lst[2],env));
                        return gv.copy(); // return value assigned to symbol
                    }
                    when "if" {
                        checkEqLstSize(lst,4);
                        if isTrue(eval(lst[1], env)) {return eval(lst[2], env);} else {return eval(lst[3], env);}
                    }
                    otherwise {
                        throw new owned ErrorWithMsg("op not implemented %t".format(op));
                    }
                }
            }
            otherwise {
                throw new owned ErrorWithMsg("undefined ast node type %t".format(ast));
            }
        }
    }

    proc test_parse(prog: string): bool {
        var f = true;
        try {
            writeln(prog);
            var tokens = tokenize(prog);
            for tok in tokens {
                write("{"); write(tok); write("}");
            }
            writeln("");
            
            writeln(parse(prog));
        }
        catch e: ErrorWithMsg {
            writeln(e.msg);
            f = false;
        }
        catch {
            writeln("unkown error!");
            f = false;
        }
        
        return f;
    }
    
    proc test_eval(prog: string): bool {
        var f = true;
        try {
            var N = 10;
            var D = {0..#N};
            var A: [D] int = D;
            var B: [D] int;

            //            writeln("entering for loop");
            // this could have the advantage of not creating array temps like the rest of arkouda does
            // forall (a,b) in zip(A,B) with (var ast = try! parse(prog3), var env = new owned Env()) {
            //forall (a,b) in zip(A,B) {
            for (a,b) in zip(A,B) {
                var ast = parse(prog); // parse and check the program
                var env = new owned Env(); // allocate the env for variables
                // addEnrtry redefines values for already existing entries
                env.addEntry("elt",a); // add a symbol called "elt" and value for a
                
                // this version does the eval the in the enviroment which creates the symbol "ans"
                //var ans = env.lookup("ans").toValue(int).v; // retrieve value for ans
                //b = ans;

                // this version just returns the GenValue from the eval call
                var ans = eval(ast,env);
                b = ans.toValue(int).v; // put answer into b
            }
            writeln(A);
            writeln(B);
        }
        catch e: ErrorWithMsg {
            writeln(e.msg);
            f = false;
        }
        catch {
            writeln("unkown error!");
            f = false;
        }

        return f;
    }

    proc test_parse_then_eval(prog: string) {
        if test_parse(prog) {test_eval(prog);} else {writeln("error!");}
    }
    
    /* test */
    proc main() {

        // very simple scheme
        // all symbols are in a predifined map

        // syntax error
        writeln(">>> Syntax error");
        var prog = "(set! ans (if (and (>= elt 5) (<= elt 5)) (+ elt 100 (- elt 10)))";
        test_parse_then_eval(prog);

        // syntax error
        writeln(">>> Syntax error");
        prog = "(set! ans (if (and (>= elt 5 (<= elt 5)) (+ elt 100) (- elt 10)))";
        test_parse_then_eval(prog);

        // eval error
        writeln(">>> Eval error: unkown symbol");
        prog = "(if (and (>= a 5) (<= elt 5)) (+ elt 100) (- elt 10))";
        test_parse_then_eval(prog);

        // eval error
        writeln(">>> Eval error: wrong numbe of args");
        prog = "(if (and (>= elt 5) (<= elt 5 1)) (+ elt 100) (- elt 10))";
        test_parse_then_eval(prog);

        // this returns the answer from the eval and also sets "ans" in the env
        writeln(">>> ans symbol");
        prog = "(set! ans (if (and (>= elt 5) (<= elt 5)) (+ elt 100) (- elt 10)))";
        test_parse_then_eval(prog);

        // this one only returns the answer from the eval
        writeln(">>> val returned from eval");
        prog = "(if (and (>= elt 5) (<= elt 5)) (+ elt 100) (- elt 10))";
        test_parse_then_eval(prog);

    }
}
