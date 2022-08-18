import json
import inspect
import ast

import numpy as np

from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray, pdarray
from arkouda.dtypes import numeric_scalars


class ArkoudaVisitor(ast.NodeVisitor):

    # initialize return string which will contain scheme/lisp code
    def __init__(self, annotations=None, args=None, echo=False):
        self.name = ""
        self.ret = ""
        self.formal_arg = {}
        self.echo = echo
        self.annotations = annotations
        self.args = args

    # allowed nodes without specialized visitors
    ALLOWED = tuple()
    ALLOWED += (ast.Module,)
    ALLOWED += (ast.Expr,)
    
    # Binary ops
    ALLOWED += (ast.BinOp,)
    def visit_BinOp(self, node):
        self.ret += " ("
        self.visit(node.op)
        self.visit(node.left)
        self.visit(node.right)
        self.ret += " )"
        if self.echo: print(self.ret)

    ALLOWED += (ast.Add,)
    def visit_Add(self, node):
        self.ret += " +"

    ALLOWED += (ast.Sub,)
    def visit_Sub(self, node):
        self.ret += " -"

    ALLOWED += (ast.Mult,)
    def visit_Mult(self, node):
        self.ret += " *"
        
    # Comparison ops
    ALLOWED += (ast.Compare,)
    def visit_Compare(self, node):
        if len(node.ops) != 1:
            raise Exception("only one comparison operator allowed")
        self.ret += " ("
        self.visit(node.ops[0]) 
        self.visit(node.left)  # left?
        self.visit(node.comparators[0]) # right?
        self.ret += " )"
        if self.echo: print(self.ret)

    ALLOWED += (ast.Eq,)
    def visit_Eq(self, node):
        self.ret += " =="

    ALLOWED += (ast.NotEq,)
    def visit_NotEq(self, node):
        self.ret += " !="

    ALLOWED += (ast.Lt,)
    def visit_Lt(self, node):
        self.ret += " <"

    ALLOWED += (ast.LtE,)
    def visit_LtE(self, node):
        self.ret += " <="

    ALLOWED += (ast.Gt,)
    def visit_Gt(self, node):
        self.ret += " >"

    ALLOWED += (ast.GtE,)
    def visit_GtE(self, node):
        self.ret += " >="

    # Boolean Ops
    ALLOWED += (ast.BoolOp,)
    def visit_BoolOp(self, node):
        self.ret += " ("
        self.visit(node.op)
        for v in node.values:
            self.visit(v)
        self.ret += " )"
        if self.echo: print(self.ret)

    ALLOWED += (ast.And,)
    def visit_And(self, node):
        self.ret += " and"

    ALLOWED += (ast.Or,)
    def visit_Or(self, node):
        self.ret += " or"

    # Unary ops (only `not` at this point)
    ALLOWED += (ast.UnaryOp,)
    def visit_UnaryOp(self, node):
        self.ret += " ("
        self.visit(node.op) 
        self.visit(node.operand)
        self.ret += " )"
        if self.echo: print(self.ret)

    ALLOWED += (ast.Not,)
    def visit_Not(self, node):
        self.ret += " not"

    # If Expression `(body) if (test) else (orelse)`
    ALLOWED += (ast.IfExp,)
    def visit_IfExp(self, node):
        self.ret += " ( if"
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)
        self.ret += " )"
        if self.echo: print(self.ret)

    # Assignment which returns a value `(x := 5)`
    ALLOWED += (ast.NamedExpr,)
    def visit_NamedExpr(self, node):
        self.ret += " ( :="
        self.visit(node.target)
        self.visit(node.value)
        self.ret += " )"
        if self.echo: print(self.ret)
        
    # Constant
    ALLOWED += (ast.Constant,)
    def visit_Constant(self, node):
        self.ret += " " + str(node.value)
        if self.echo: print(self.ret)

    # Name, mostly variable names, I think
    ALLOWED += (ast.Name,)
    def visit_Name(self, node):
        self.ret += " " + node.id
        if self.echo: print(self.ret)
        
    # argument name in argument list
    ALLOWED += (ast.arg,)
    def visit_arg(self, node, i):
        self.formal_arg[node.arg] = node.annotation.attr
        if isinstance(self.args[i],pdarray): # need to have same size for all
            self.ret += " ( := " + node.arg + " ( lookup_and_index " + self.args[i].name + " i ) ) "
        elif isinstance(self.args[i], numeric_scalars):
            self.ret += " ( := " + node.arg + " " + str(self.args[i]) + " ) "
        else:
            raise Exception("unhandled arg type = " + str(self.args[i].type))
        if self.echo: print(self.ret)
        
    # argument list
    ALLOWED += (ast.arguments,)
    def visit_arguments(self, node):
        for (i, a) in enumerate(node.args):
            self.visit_arg(a, i)
        if self.echo: print(self.ret)

    # Return, `return value`
    ALLOWED += (ast.Return,)
    def visit_Return(self, node):
        self.ret += " ( return"
        self.visit(node.value)
        self.ret += " )"
        if self.echo: print(self.ret)

    # Function Definition
    ALLOWED += (ast.FunctionDef,)
    def visit_FunctionDef(self, node):
        self.name = node.name
        self.ret += "( begin"
        self.visit_arguments(node.args)
        for b in node.body:
            self.visit(b)
        self.ret += " )"
        if self.echo: print(self.ret)

    # override ast.NodeVisitor.visit() method
    def visit(self, node):
        """Ensure only contains allowed ast nodes."""
        if not isinstance(node, self.ALLOWED):
            raise SyntaxError("ast node not allowed: " + str(node))
        ast.NodeVisitor.visit(self, node)



# Arkouda function decorator
# transforms a simple python function into a Scheme/Lisp lambda
# which could be sent to the arkouda server to be evaluated there
def arkouda_func(func):
    def wrapper(*args):
        num_elems = -1
        elem_dtypes = []
        for arg in args:
            if isinstance(arg, pdarray):
                if num_elems == -1:
                    num_elems = arg.size
                    elem_dtypes.append(arg.dtype)
                else:
                    if arg.size != num_elems:
                        raise Exception("size mismatch exception; all pdarrays must be same size")
        
         # get source code for function
        source_code = inspect.getsource(func)
        # print("source_code :\n" + source_code)

         # parse sorce code into a python ast
        tree = ast.parse(source_code)
        #print(ast.dump(tree, indent=4))

         # create ast visitor to transform ast into scheme/lisp code
        visitor = ArkoudaVisitor(annotations=func.__annotations__, args=args, echo=False)
        visitor.visit(tree)

        # construct a message to the arkouda server
        # send it
        # get result
        # return pdarray of result
        repMsg = generic_msg(cmd="lispCode", args=f"float64 | {num_elems} | {visitor.ret}")
        
        # return a dummy pdarray
        return create_pdarray(repMsg)
        
    return wrapper


