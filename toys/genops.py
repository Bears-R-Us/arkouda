#!/usr/bin/env python3

from string import Template
import os

OPEQUAL = ['+=', '-=', '*=', '/=']
BINOP = ['+', '-', '*', '/']

opeqGenSym = Template("""
    proc ${opeq}(left : GenSymEntry, right : GenSymEntry) {
        select (left.dtype, right.dtype) {
            when (DType.Int64, DType.Int64) {
                var leftInt64 = left: SymEntry(int);
                var rightInt64 = right: SymEntry(int);
                leftInt64.a ${opeq} rightInt64.a;
            }
            when (DType.Int64, DType.Float64) {
                var leftInt64 = left: SymEntry(int);
                var rightFloat64 = right: SymEntry(real);
                leftInt64.a ${opeq} rightFloat64.a:int;
            }
            when (DType.Float64, DType.Int64) {
                var leftFloat64 = left: SymEntry(real);
                var rightInt64 = right: SymEntry(int);
                leftFloat64.a ${opeq} rightInt64.a:real;
            }
            when (DType.Float64, DType.Float64) {
                var leftFloat64 = left: SymEntry(real);
                var rightFloat64 = right: SymEntry(real);
                leftFloat64.a ${opeq} rightFloat64.a;
            }
            otherwise { writeln("should never happen"); }
        }
    }
""")

binopGenSym = Template("""
    proc ${binop}(left : GenSymEntry, right : GenSymEntry): shared GenSymEntry {
        select (left.dtype, right.dtype) {
            when (DType.Int64, DType.Int64) {
                var leftInt64 = left: SymEntry(int);
                var rightInt64 = right: SymEntry(int);
                var a = leftInt64.a ${binop} rightInt64.a;
                return new shared SymEntry(a);
            }
            when (DType.Int64, DType.Float64) {
                var leftInt64 = left: SymEntry(int);
                var rightFloat64 = right: SymEntry(real);
                var a = leftInt64.a ${binop} rightFloat64.a;
                return new shared SymEntry(a);
            }
            when (DType.Float64, DType.Int64) {
                var leftFloat64 = left: SymEntry(real);
                var rightInt64 = right: SymEntry(int);
                var a = leftFloat64.a ${binop} rightInt64.a;
                return new shared SymEntry(a);
            }
            when (DType.Float64, DType.Float64) {
                var leftFloat64 = left: SymEntry(real);
                var rightFloat64 = right: SymEntry(real);
                var a = leftFloat64.a ${binop} rightFloat64.a;
                return new shared SymEntry(a);
            }
            otherwise { writeln("should never happen"); }
        }
        return nil; // undefined!! should never happen 
    }
""")

opeqScalar = Template("""
    proc ${opeq}(left: GenSymEntry, scalar : int) {
        select (left.dtype) {
            when (DType.Int64) {
                var leftInt64 = left: SymEntry(int);
                leftInt64.a ${opeq} scalar;
            }
            when (DType.Float64) {
                var leftFloat64 = left: SymEntry(real);
                leftFloat64.a ${opeq} scalar:real;
            }
            otherwise { writeln("should never happen"); }
        }
    }
    
    proc ${opeq}(left: GenSymEntry, scalar : real) {
        select (left.dtype) {
            when (DType.Int64) {
                var leftInt64 = left: SymEntry(int);
                leftInt64.a ${opeq} scalar:int;
            }
            when (DType.Float64) {
                var leftFloat64 = left: SymEntry(real);
                leftFloat64.a ${opeq} scalar;
            }
            otherwise { writeln("should never happen"); }
        }
    }
""")

binopScalar = Template("""
    proc ${binop}(left: GenSymEntry, scalar : int): shared GenSymEntry {
        select (left.dtype) {
            when (DType.Int64) {
                var leftInt64 = left: SymEntry(int);
                var a = leftInt64.a ${binop} scalar;
                return new shared SymEntry(a);
            }
            when (DType.Float64) {
                var leftFloat64 = left: SymEntry(real);
                var a = leftFloat64.a ${binop} scalar:real;
                return new shared SymEntry(a);
            }
            otherwise { writeln("should never happen"); }
        }
        return nil; // undefined!! should never happen 
    }
    
    proc ${binop}(left: GenSymEntry, scalar : real): shared GenSymEntry {
        select (left.dtype) {
            when (DType.Int64) {
                var leftInt64 = left: SymEntry(int);
                var a = leftInt64.a ${binop} scalar:int; // need cast to int
                return new shared SymEntry(a);
            }
            when (DType.Float64) {
                var leftFloat64 = left: SymEntry(real);
                var a = leftFloat64.a ${binop} scalar;
                return new shared SymEntry(a);
            }
            otherwise { writeln("should never happen"); }
        }
        return nil; // undefined!! should never happen 
    }
    
    proc ${binop}(scalar : int, right: GenSymEntry): shared GenSymEntry {
        select (right.dtype) {
            when (DType.Int64) {
                var rightInt64 = right: SymEntry(int);
                var a = scalar ${binop} rightInt64.a;
                return new shared SymEntry(a);
            }
            when (DType.Float64) {
                var rightFloat64 = right: SymEntry(real);
                var a = scalar:real ${binop} rightFloat64.a;
                return new shared SymEntry(a);
            }
            otherwise { writeln("should never happen"); }
        }
        return nil; // undefined!! should never happen 
    }
    
    proc ${binop}(scalar : real, right: GenSymEntry): shared GenSymEntry {
        select (right.dtype) {
            when (DType.Int64) {
                var rightInt64 = right: SymEntry(int);
                var a = scalar ${binop} rightInt64.a; // need cast to int
                return new shared SymEntry(a);
            }
            when (DType.Float64) {
                var rightFloat64 = right: SymEntry(real);
                var a = scalar ${binop} rightFloat64.a;
                return new shared SymEntry(a);
            }
            otherwise { writeln("should never happen"); }
        }
        return nil; // undefined!! should never happen 
    }
""")

modTemplate = Template("""// this module was generated
module ${name}
{
    use MultiTypeSymEntry;
${body}
}
""")

def gen_module(name):
    body = [opeqGenSym.substitute(opeq=op) for op in OPEQUAL]
    body += [binopGenSym.substitute(binop=op) for op in BINOP]
    body += [opeqScalar.substitute(opeq=op) for op in OPEQUAL]
    body += [binopScalar.substitute(binop=op) for op in BINOP]
    return modTemplate.substitute(name=name, body='\n'.join(body))

def write_module(path):
    name, ext = os.path.splitext(os.path.basename(path))
    with open(path, 'w') as f:
        f.write(gen_module(name))

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} output_filename")
        sys.exit()
    write_module(sys.argv[1])
