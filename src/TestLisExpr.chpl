module TestLisExpr
{

    use LisExprData;
    use LisExprInterp;
    
    /* parse expression and return true if no parse errors */
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
        catch e: Error {
            writeln(e.message());
            f = false;
        }
       
        return f;
    }

    /* eval with a specific environment for the test expression return true if no errors */
    proc test_eval(prog: string): bool {
        var f = true;
        try {
            var N = 10;
            var D = {0..#N};
            var A: [D] int = D;
            var B: [D] int;
            
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
        catch e: Error {
            writeln(e.message());
            f = false;
        }

        return f;
    }

     // tasks per locale based on locale0
    const numTasks = here.maxTaskPar;
    const Tasks = {0..#numTasks}; // these need to be const for comms/performance reasons
    
    // calculate sub-domain for task
    inline proc calcBlock(task: int, low: int, high: int) {
        var totalsize = high - low + 1;
        var div = totalsize / numTasks;
        var rem = totalsize % numTasks;
        var rlow: int;
        var rhigh: int;
        if (task < rem) {
            rlow = task * (div+1) + low;
            rhigh = rlow + div;
        }
        else {
            rlow = task * div + rem + low;
            rhigh = rlow + div - 1;
        }
        return {rlow .. rhigh};
    }

    /* eval with a specific environment for the test expression return true if no errors */
    proc test_eval_coforall(const in prog: string): bool {
        var f = true;
        var N = 10;
        var D = {0..#N};
        var A: [D] int = D;
        var B: [D] int;
        
        // this could have the advantage of not creating array temps like the rest of arkouda does
        try {
            coforall loc in Locales {
                on loc {
                    coforall task in Tasks {
                        // get local domain's indices
                        var lD = A.domain.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
                        
                        // parse and check the program
                        var ast = parse(prog);
                        // allocate the env for variables
                        var env = new owned Env();
                        
                        for i in tD {
                            // addEnrtry redefines values for already existing entries
                            env.addEntry("elt",A[i]); // add a symbol called "elt" and value for a
                            
                            // this version does the eval the in the enviroment which creates the symbol "ans"
                            //var ans = env.lookup("ans").toValue(int).v; // retrieve value for ans
                            //b = ans;
                            
                            // this version just returns the GenValue from the eval call
                            var ans = eval(ast,env);
                            B[i] = ans.toValue(int).v; // put answer into b
                        }
                    }
                }
                
                writeln(A);
                writeln(B);
            }
        }
        catch e: Error {
          if e.type == TaskErrors {
            for err in e {
              writeln(err.message());
            }
          } else {
            writeln(e.message());
          }
          f = false;
        }
        
        return f;
    }

    proc test_parse_then_eval(prog: string) {
        if test_parse(prog) {test_eval(prog);} else {writeln("error!");}
    }
    
    proc parallel_test_parse_then_eval(prog: string) {
        if test_parse(prog) {test_eval_coforall(prog);} else {writeln("error!");}
    }
    
    /* test */
    proc main() {

        var prog: string;
        
        // very simple scheme
        // all symbols are in a predifined map

        //
        // Serial eval
        //
        
       // syntax error
        writeln(">>> Syntax error");
        prog = "(:= ans (if (and (>= elt 5) (<= elt 5)) (+ elt 100 (- elt 10)))";
        test_parse_then_eval(prog);

        // syntax error
        writeln(">>> Syntax error");
        prog = "(:= ans (if (and (>= elt 5 (<= elt 5)) (+ elt 100) (- elt 10)))";
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
        prog = "(:= ans (if (and (>= elt 5) (<= elt 5)) (+ elt 100) (- elt 10)))";
        test_parse_then_eval(prog);

        // this one only returns the answer from the eval
        writeln(">>> val returned from eval");
        prog = "(if (and (>= elt 5) (<= elt 5)) (+ elt 100) (- elt 10))";
        test_parse_then_eval(prog);

        //
        // Parallel eval
        //
        
       // syntax error
        writeln(">>> Syntax error");
        prog = "(:= ans (if (and (>= elt 5) (<= elt 5)) (+ elt 100 (- elt 10)))";
        parallel_test_parse_then_eval(prog);

        // syntax error
        writeln(">>> Syntax error");
        prog = "(:= ans (if (and (>= elt 5 (<= elt 5)) (+ elt 100) (- elt 10)))";
        parallel_test_parse_then_eval(prog);

        // eval error
        writeln(">>> Eval error: unkown symbol");
        prog = "(if (and (>= a 5) (<= elt 5)) (+ elt 100) (- elt 10))";
        parallel_test_parse_then_eval(prog);

        // eval error
        writeln(">>> Eval error: wrong numbe of args");
        prog = "(if (and (>= elt 5) (<= elt 5 1)) (+ elt 100) (- elt 10))";
        parallel_test_parse_then_eval(prog);

        // this returns the answer from the eval and also sets "ans" in the env
        writeln(">>> ans symbol");
        prog = "(:= ans (if (and (>= elt 5) (<= elt 5)) (+ elt 100) (- elt 10)))";
        parallel_test_parse_then_eval(prog);

        // this one only returns the answer from the eval
        writeln(">>> val returned from eval");
        prog = "(if (and (>= elt 5) (<= elt 5)) (+ elt 100) (- elt 10))";
        parallel_test_parse_then_eval(prog);
        
    }
}
