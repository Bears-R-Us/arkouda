import itertools
import json
import sys
from enum import Enum

import chapel

DEFAULT_MODS = ["MsgProcessing", "GenSymIO"]

registerAttr = ("arkouda.registerCommand", ["name"])
instAndRegisterAttr = ("arkouda.instantiateAndRegister", ["prefix"])

# chapel types and their numpy equivalents
chapel_scalar_types = {
    "string": "str",
    "bool": "bool",
    "int(8)": "int8",
    "int(16)": "int16",
    "int(32)": "int32",
    "int(64)": "int64",
    "int": "int64",
    "uint(8)": "uint8",
    "uint(16)": "uint16",
    "uint(32)": "uint32",
    "uint(64)": "uint64",
    "uint": "uint64",
    "real(32)": "float32",
    "real(64)": "float64",
    "real": "float64",
    "complex(64)": "complex64",
    "complex(128)": "complex128",
    "complex": "complex128",
    "bigint": "bigint",
}

# type and variable names from arkouda infrastructure that could conceivable be changed in the future:
ARGS_FORMAL_INTENT = "<default-intent>"
ARGS_FORMAL_NAME = "msgArgs"
ARGS_FORMAL_TYPE = "borrowed MessageArgs"

SYMTAB_FORMAL_INTENT = "<default-intent>"
SYMTAB_FORMAL_NAME = "st"
SYMTAB_FORMAL_TYPE = "borrowed SymTab"

ARRAY_ENTRY_CLASS_NAME = "SymEntry"

RESPONSE_TYPE_NAME = "MsgTuple"


def error_message(message, details, loc=None):
    if loc:
        info = str(loc).split(":")
        print(" [", info[0], ":", info[1], "] ", file=sys.stderr, end="")

    print("Error ", message, ": ", details, file=sys.stderr)


def is_registerable(fn):
    """
    Check if a function is registerable as an Arkouda command
    """
    if fn.is_method():
        error_message(
            f"registering {fn.name()}",
            "cannot register methods as commands",
            fn.location(),
        )
        return False
    return True


def annotated_procs(root, attr):
    """
    Find all procs in root annotated with a particular attribute
    """
    for fn, _ in chapel.each_matching(root, chapel.Function):
        if ag := fn.attribute_group():
            for a in ag:
                if attr_call := chapel.parse_attribute(a, attr):
                    if is_registerable(fn):
                        yield fn, attr_call


def extract_ast_block_text(node):
    loc = node.location()
    lstart, cstart = loc.start()
    lstop, cstop = loc.end()

    with open(loc.path(), "r") as f:
        # get to the start line
        for i in range(int(lstart) - 1):
            f.readline()

        # get to the start column
        f.read(int(cstart) - 1)

        # read the block text
        return f.read(int(cstop) - int(cstart)).strip()


class FormalQuery:
    def __init__(self, name):
        self.name = name

    def name(self):
        return self.name


class FormalQueryRef:
    def __init__(self, name):
        self.name = name

    def name(self):
        return self.name


class StaticTypeInfo:
    def __init__(self, value):
        self.value = value

    def value(self):
        return self.value


class formalKind(Enum):
    ARRAY = 1
    LIST = 2
    BORROWED_CLASS = 3
    HOMOG_TUPLE = 4
    SCALAR = 5


class FormalTypeSpec:
    def __init__(self, kind, name, storage_kind, type_str=None, info=None):
        self.kind = kind
        self.name = name
        self.storage_kind = storage_kind
        self.type_str = type_str
        self.info = (
            []
            if info is None
            else (
                info
                if isinstance(info, list)
                else [
                    info,
                ]
            )
        )

    def append_info(self, info):
        self.info.append(info)

    def info(self):
        return self.info

    def is_generic(self) -> bool:
        return self.storage_kind in ["type", "param"]

    def is_untyped(self) -> bool:
        return self.kind == formalKind.SCALAR and self.type_str is None

    def is_chapel_scalar_type(self) -> bool:
        return self.kind == formalKind.SCALAR and self.type_str in chapel_scalar_types

    def stringify(self) -> str:
        return (
            f"{self.storage_kind} {self.name}: {self.type_str}"
            if self.type_str
            else f"{self.storage_kind} {self.name}"
        )


def get_formals(fn, require_type_annotations):
    """
    Get a function's formal parameters, separating them into concrete and
    generic (param and type) formals
    The (name, storage kind, type expression, info) of each formal is returned
    Info is used to store extra information about the formal type, such as the
    presence of a type-query in a bracket-loop type expression
    """

    def get_formal_type_spec(formal):
        name = formal.name()
        sk = formal.storage_kind()
        if te := formal.type_expression():
            if isinstance(te, chapel.BracketLoop):  # Array types
                if te.is_maybe_array_type():
                    ft = FormalTypeSpec(formalKind.ARRAY, name, sk)

                    # record domain query name, if any
                    if isinstance(te.iterand(), chapel.TypeQuery):
                        dom_q = FormalQuery(te.iterand().name())
                    else:
                        # hack: chapel-py doesn't have a way to get the text/body of a block (I think)
                        block_text = extract_ast_block_text(te.iterand())
                        dom_q = FormalQueryRef(block_text)
                    ft.append_info(dom_q)

                    # record element type query, if any
                    elt_q = None
                    if isinstance(te.body(), chapel.Block):
                        # hack: chapel-py doesn't have a way to get the text/body of a block (I think)
                        block_text = extract_ast_block_text(te.body())

                        if block_text[0] == "?":
                            elt_q = FormalQuery(block_text[1:])
                        elif block_text in chapel_scalar_types.keys():
                            elt_q = StaticTypeInfo(block_text)
                        else:
                            elt_q = FormalQueryRef(block_text)
                    ft.append_info(elt_q)

                    return ft
                else:
                    # TODO:  `x: []` and `x: [?d]` are currently treated as invalid formal type expressions
                    raise ValueError("invalid formal type expression")
            elif isinstance(
                te, chapel.FnCall
            ):  # Composite types (list and borrowed-class)
                if ce := te.called_expression():
                    if isinstance(ce, chapel.Identifier):
                        call_name = ce.name()
                        if call_name == "list":
                            list_type = list(te.actuals())[0].name()
                            if list_type not in chapel_scalar_types:
                                error_message(
                                    f"registering '{fn.name()}'",
                                    f"unsupported formal type for registration {list_type}; list element type must be a scalar",
                                )
                            else:
                                return FormalTypeSpec(
                                    formalKind.LIST, name, sk, info=list_type
                                )
                        elif call_name == "borrowed":
                            ten = call_name + " " + list(te.actuals())[0].name()
                            return FormalTypeSpec(
                                formalKind.BORROWED_CLASS, name, sk, ten
                            )
                        else:
                            error_message(
                                f"registering '{fn.name()}'",
                                f"unsupported composite type expression for registration {call_name}",
                            )
                    else:
                        error_message(
                            f"registering '{fn.name()}'",
                            f"unsupported type expression for registration {extract_ast_block_text(te.body())}",
                        )
                else:
                    error_message(
                        f"registering '{fn.name()}'",
                        f"unsupported type expression for registration {extract_ast_block_text(te.body())}",
                    )
            elif (
                isinstance(te, chapel.OpCall) and te.is_binary_op() and te.op() == "*"
            ):  # Homog. Tuple types
                ft = FormalTypeSpec(formalKind.HOMOG_TUPLE, name, sk)
                actuals = list(te.actuals())

                # check the tuple size (should be an int literal or a queried domain's rank)
                if isinstance(actuals[0], chapel.IntLiteral):
                    ft.append_info(StaticTypeInfo(int(actuals[0].text())))
                elif (
                    isinstance(actuals[0], chapel.Dot)
                    and actuals[0].field() == "rank"
                    and isinstance(actuals[0].receiver(), chapel.Identifier)
                ):
                    ft.append_info(FormalQueryRef(actuals[0].receiver().name()))
                else:
                    error_message(
                        f"registering '{fn.name()}'",
                        f"unsupported homog_tuple type expression for registration on formal '{formal.name()}'; "
                        + "tuple size must be an int literal or a queried domain's rank",
                    )

                # check the tuple element type (must be an identifier - either corresponding to a scalar type or a dtype query)
                if isinstance(actuals[1], chapel.Identifier):
                    name = actuals[1].name()
                    if name in chapel_scalar_types:
                        ft.append_info(StaticTypeInfo(name))
                    else:
                        ft.append_info(FormalQueryRef(name))
                else:
                    error_message(
                        f"registering '{fn.name()}'",
                        f"unsupported homog_tuple type expression for registration on formal '{formal.name()}'; "
                        + f"tuple element type must be an identifier (either a Chapel scalar type or a queried dtype)",
                    )

                return ft
            else:  # Scalar types
                return FormalTypeSpec(formalKind.SCALAR, name, sk, te.name())
        else:  # param and type formals
            return FormalTypeSpec(formalKind.SCALAR, name, sk)

    con_formals = []
    gen_formals = []
    for formal in fn.formals():
        if isinstance(formal, (chapel.TupleDecl, chapel.VarArgFormal)):
            raise ValueError(
                "registration of procedures with vararg or tuple-grouped formals are not yet supported."
            )
        elif isinstance(formal, (chapel.Formal, chapel.AnonFormal)):
            spec = get_formal_type_spec(formal)

            if spec.is_untyped() and require_type_annotations:
                error_message(
                    f"registering '{fn.name()}'",
                    f"missing type expression for formal '{spec.name}'",
                )
            else:
                if spec.is_generic():
                    gen_formals.append(spec)
                else:
                    con_formals.append(spec)

    return con_formals, gen_formals


def clean_stamp_name(name):
    """
    Remove any invalid characters from a stamped command name
    """
    return name.translate(str.maketrans("[](),=", "______"))


def stamp_generic_command(
    generic_proc_name, prefix, module_name, formals, line_num, is_user_proc
):
    """
    Create code to stamp out and register a generic command using a generic
    procedure, and a set values for its generic formals.

    formals should be in the format {p1: v1, p2: v2, ...}

    the stamped command will have a name in the format: prefix<v1, v2, ...>
    """

    command_name = (
        prefix
        + "<"
        + ",".join(
            [
                # if the generic formal is a 'type' convert it to its numpy dtype name
                (chapel_scalar_types[v] if v in chapel_scalar_types else str(v))
                for _, v in formals.items()
            ]
        )
        + ">"
    )

    stamp_name = f"ark_{clean_stamp_name(prefix)}_" + "_".join(
        [str(v).replace("(", "").replace(")", "") for _, v in formals.items()]
    )

    stamp_formal_args = ", ".join([f"{k}={v}" for k, v in formals.items()])

    # use qualified naming if generic_proc belongs in a use defined module to avoid name conflicts
    call = f"{module_name}.{generic_proc_name}" if is_user_proc else generic_proc_name

    proc = (
        f"proc {stamp_name}(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): {RESPONSE_TYPE_NAME} throws do\n"
        + f"  return {call}(cmd, msgArgs, st, {stamp_formal_args});\n"
        + f"registerFunction('{command_name}', {stamp_name}, '{module_name}', {line_num});"
    )
    return proc


def permutations(to_permute):
    """
    Take an object in the format {p1: [v1, v2, ...], p2: [w1, w2, ...], ...}

    and return a list of permutations of all the values of the parameters in
    the format:

    [{p1: v1, p2: w1, ...}, {p1: v1, p2: w2, ...}, ...]
    """
    perms = [{}]
    for parameter, values in to_permute.items():
        new_perms = []
        for p in perms:
            for v in values:
                x = p.copy()
                x.update({parameter: v})
                new_perms.append(x)
        perms = new_perms
    return perms


# TODO: validate the 'parameter_classes' fields at the beginning of this script
# s.t. invalid values don't emit an error message associated with a specific proc
# (i.e., error instantiating 'meanMsg' - invalid parameter value type (list) in list '[["a", "b"], "c"]')
def parse_param_class_value(value):
    """
    Parse a value from the 'parameter_classes' field in the configuration file

    Allows scalars, lists of scalars, or strings that can be evaluated as lists
    """
    if isinstance(value, list):
        for v in value:
            if not isinstance(v, (int, float, str)):
                raise ValueError(
                    f"Invalid parameter value type ({type(v)}) in list '{value}'"
                )
        return value
    elif isinstance(value, int):
        return [
            value,
        ]
    elif isinstance(value, float):
        return [
            value,
        ]
    elif isinstance(value, str):
        # evaluate string as python code, resulting in a list
        vals = eval(value)
        if isinstance(vals, list):
            return vals
        else:
            raise ValueError(
                f"Could not create a list of parameter values from '{value}'"
            )
    else:
        raise ValueError(f"Invalid parameter value type ({type(value)}) for '{value}'")


def generic_permutations(config, gen_formals):
    """
    Create a list of all possible permutations of the generic formals, using
    values from the 'parameter_classes' field in the configuration file
    """
    to_permute = {}
    # TODO: check that the type annotations on param formals are scalars and that
    # the values in the config file can be used as literals of that type
    for formal_spec in gen_formals:
        name_components = formal_spec.name.split("_")

        if len(name_components) < 2:
            raise ValueError(
                f"invalid generic formal '{formal_spec.name}; "
                + "generic formals must be in the format '<param-class>_<param-name>[_...]' "
                + "to be instantiated with values from the configuration file"
            )

        pclass = name_components[0]  # ex: 'array'
        pname = name_components[1]  # ex: 'nd'

        if pclass not in config["parameter_classes"].keys():
            raise ValueError(
                f"generic formal '{formal_spec.name}' is not a valid parameter class; "
                + "please check the 'parameter_classes' field in the configuration file"
            )

        if pname not in config["parameter_classes"][pclass].keys():
            raise ValueError(
                f"parameter class '{pclass}' has no parameter '{pname}'; "
                + "please check the 'parameter_classes' field in the configuration file"
            )

        to_permute[formal_spec.name] = parse_param_class_value(
            config["parameter_classes"][pclass][pname]
        )

    return permutations(to_permute)


def valid_generic_command_signature(fn, con_formals):
    """
    Ensure that a proc's signature matches the format:
    'proc <name>(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws'
    s.t. it's instantiations can be registered in the command table
    """
    if len(con_formals) == 3:
        if (
            con_formals[0].name != "cmd"
            or con_formals[0].type_str != "string"
            or con_formals[1].name != ARGS_FORMAL_NAME
            or con_formals[1].storage_kind != ARGS_FORMAL_INTENT
            or con_formals[1].type_str != ARGS_FORMAL_TYPE
            or con_formals[2].name != SYMTAB_FORMAL_NAME
            or con_formals[2].storage_kind != SYMTAB_FORMAL_INTENT
            or con_formals[2].type_str != SYMTAB_FORMAL_TYPE
        ):
            return False
    else:
        return False

    if rt := fn.return_type():
        if rt.name() != RESPONSE_TYPE_NAME:
            return False
    else:
        return False

    if not fn.throws():
        return False

    return True


# TODO: use var/const depending on user proc's formal intent
def unpack_array_arg(arg_name, array_count, finfo, domain_queries, dtype_queries):
    """
    Generate the code to unpack an array symbol from the symbol table

    'array_count' is used to generate unique names for the dtype and nd arguments when
    a procedure has multiple array-symbol formals

    Example:
    ```
    var x_array_sym = st[msgArgs['x']]: SymEntry(array_dtype_0, array_nd_0);
    ref x = x_array_sym.a;
    ```

    Returns the chapel code, and the specifications for the
    'etype' and 'dimensions' type-constructor arguments
    """

    # check if the nd formal is a domain query
    if (
        finfo is not None
        and finfo[0] is not None
        and isinstance(finfo[0], FormalQueryRef)
        and finfo[0].name in domain_queries
    ):
        nd_arg_name = domain_queries[finfo[0].name]
        nd_generic_formal_info = None
    else:
        nd_arg_name = "array_nd_" + str(array_count)
        nd_generic_formal_info = FormalTypeSpec(
            formalKind.SCALAR, nd_arg_name, "param", "int"
        )

    # check if the array formal has a static type or a type-query
    # if not, generate a unique name and formal info for the dtype argument
    if (
        finfo is not None
        and finfo[1] is not None
        and isinstance(finfo[1], StaticTypeInfo)
    ):
        dtype_arg_name = finfo[1].value
        dtype_generic_formal_info = None
    elif (
        finfo is not None
        and finfo[1] is not None
        and isinstance(finfo[1], FormalQueryRef)
        and finfo[1].name in dtype_queries
    ):
        dtype_arg_name = dtype_queries[finfo[1].name]
        dtype_generic_formal_info = None
    else:
        dtype_arg_name = "array_dtype_" + str(array_count)
        dtype_generic_formal_info = FormalTypeSpec(
            formalKind.SCALAR, dtype_arg_name, storage_kind="type"
        )

    return (
        f"\tvar {arg_name}_array_sym = {SYMTAB_FORMAL_NAME}[{ARGS_FORMAL_NAME}['{arg_name}']]: {ARRAY_ENTRY_CLASS_NAME}({dtype_arg_name}, {nd_arg_name});\n"
        + f"\tref {arg_name} = {arg_name}_array_sym.a;",
        dtype_generic_formal_info,
        nd_generic_formal_info,
    )


def unpack_user_symbol(arg_name, symbol_class):
    """
    Generate the code to unpack a user-defined symbol from the symbol table.

    - symbol_class should inherit from 'AbstractSymEntry'
    - generic user-defined symbol types are not yet supported

    Example:
    ```
    var x = st[msgArgs['x']]: MySymbolType;
    ```
    """
    return f"\tvar {arg_name} = {SYMTAB_FORMAL_NAME}[{ARGS_FORMAL_NAME}['{arg_name}']]: {symbol_class};"


def unpack_scalar_arg(arg_name, arg_type):
    """
    Generate the code to unpack a scalar argument

    Example:
    ```
    var x = msgArgs['x'].toScalar(int);
    ```
    """
    return f"\tvar {arg_name} = {ARGS_FORMAL_NAME}['{arg_name}'].toScalar({arg_type});"


def unpack_scalar_arg_with_generic(arg_name, scalar_count):
    """
    Generate the code to unpack a scalar argument

    'scalar_count' is used to generate unique names when
    a procedure has multiple scalar-symbol formals

    Example:
    ```
    var x = msgArgs['x'].toScalar(scalar_dtype_0);
    ```

    Returns the chapel code, and the specifications for the
    'dtype' and type-constructor arguments
    """
    dtype_arg_name = "scalar_dtype_" + str(scalar_count)
    return (
        unpack_scalar_arg(arg_name, dtype_arg_name),
        [(dtype_arg_name, "type", None, None)],
    )


def unpack_tuple_arg(arg_name, tuple_size, scalar_type):
    """
    Generate the code to unpack a tuple argument

    Example:
    ```
    var x = msgArgs['x'].getTuple(int, 2);
    ```
    """
    return f"\tvar {arg_name} = {ARGS_FORMAL_NAME}['{arg_name}'].toScalarTuple({scalar_type}, {tuple_size});"


def unpack_list_arg(arg_name, list_elt_type):
    """
    Generate the code to unpack a list argument

    Example:
    ```
    var x = msgArgs['x'].toScalarList(int);
    ```
    """

    return f"\tvar {arg_name} = {ARGS_FORMAL_NAME}['{arg_name}'].toScalarList({list_elt_type});"


# TODO: also support generic user-defined symbol types, not just arrays
def gen_signature(user_proc_name, generic_args=None):
    """
    Generate the signature for a message handler procedure

    For a concrete command procedure:
    ```
    proc ark_reg_<user_proc_name>(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    ```

    For a generic command procedure:
    ```
    proc ark_reg_<user_proc_name>_generic(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, <generic args>): MsgTuple throws {
    ```

    Return the signature and the name of the procedure
    """
    if generic_args:
        name = "ark_reg_" + user_proc_name + "_generic"
        arg_strings = [formal_spec.stringify() for formal_spec in generic_args]
        proc = f"proc {name}(cmd: string, {ARGS_FORMAL_NAME}: {ARGS_FORMAL_TYPE}, {SYMTAB_FORMAL_NAME}: {SYMTAB_FORMAL_TYPE}, {', '.join(arg_strings)}): {RESPONSE_TYPE_NAME} throws {'{'}"
    else:
        name = "ark_reg_" + user_proc_name
        proc = f"proc {name}(cmd: string, {ARGS_FORMAL_NAME}: {ARGS_FORMAL_TYPE}, {SYMTAB_FORMAL_NAME}: {SYMTAB_FORMAL_TYPE}): {RESPONSE_TYPE_NAME} throws {'{'}"
    return (proc, name)


def gen_arg_unpacking(formals):
    """
    Generate argument unpacking code for a message handler procedure

    Returns the chapel code to unpack the arguments, and a list of generic arguments
    """
    unpack_lines = []
    generic_args = []
    array_arg_counter = 0
    scalar_arg_counter = 0

    array_domain_queries = {}
    array_dtype_queries = {}

    for formal_spec in formals:
        if formal_spec.is_chapel_scalar_type():
            unpack_lines.append(
                unpack_scalar_arg(formal_spec.name, formal_spec.type_str)
            )
        elif formal_spec.kind == formalKind.ARRAY:
            # finfo[0] is the domain query info, finfo[1] is the dtype query info
            finfo = formal_spec.info

            code, gen_dtype_arg, gen_nd_arg = unpack_array_arg(
                formal_spec.name,
                array_arg_counter,
                finfo,
                array_domain_queries,
                array_dtype_queries,
            )
            unpack_lines.append(code)
            if gen_dtype_arg is not None:
                generic_args.append(gen_dtype_arg)
            if gen_nd_arg is not None:
                generic_args.append(gen_nd_arg)
            array_arg_counter += 1

            # When an array formal type has a domain query (e.g., '[?d]'), keep track of
            # the array's generic rank argument under the domain query's name (e.g., 'd').
            # this allows homogeneous-tuple formal types to use the array's rank as a size argument
            # Do the same for dtype queries
            if finfo is not None:
                if (
                    finfo[0] is not None
                    and isinstance(finfo[0], FormalQuery)
                    and gen_nd_arg is not None
                ):
                    array_domain_queries[finfo[0].name] = gen_nd_arg.name
                if (
                    finfo[1] is not None
                    and isinstance(finfo[1], FormalQuery)
                    and gen_dtype_arg is not None
                ):
                    array_dtype_queries[finfo[1].name] = gen_dtype_arg.name

        elif formal_spec.kind == formalKind.LIST:
            unpack_lines.append(unpack_list_arg(formal_spec.name, formal_spec.info[0]))
        elif formal_spec.kind == formalKind.HOMOG_TUPLE:
            finfo = formal_spec.info
            tsize = finfo[0]
            ttype = finfo[1]

            # if the tuple size is a domain query, use the corresponding generic rank argument
            if isinstance(tsize, FormalQueryRef):
                tsize = array_domain_queries[tsize.name]
            else:
                tsize = tsize.value

            # if the tuple type is a dtype query, use the corresponding generic dtype argument
            if isinstance(ttype, FormalQueryRef) and ttype.name in array_dtype_queries:
                ttype = array_dtype_queries[ttype]
            else:
                ttype = ttype.value

            unpack_lines.append(unpack_tuple_arg(formal_spec.name, tsize, ttype))
        else:
            # a scalar formal with a generic type
            if formal_spec.type_str is not None:
                if queried_type := array_dtype_queries[formal_spec.type_str]:
                    unpack_lines.append(
                        unpack_scalar_arg(formal_spec.name, queried_type)
                    )
                else:
                    # TODO: fully handle generic user-defined types
                    code, scalar_args = unpack_scalar_arg_with_generic(
                        formal_spec.name, scalar_arg_counter
                    )
                    unpack_lines.append(code)
                    generic_args += scalar_args
                    scalar_arg_counter += 1

    return ("\n".join(unpack_lines), generic_args)


def gen_user_function_call(name, arg_names, mod_name, user_rt):
    """
    Generate code to call a user-defined function with the given arguments

    Examples:
    ```
    var ark_result = MyModule.myProc(x, y, z);
    ```

    Returns (chapel code, result name)
    """
    if user_rt:
        return (
            f"\tvar ark_result = {mod_name}.{name}({','.join(arg_names)});",
            "ark_result",
        )
    else:
        return (f"\t{mod_name}.{name}({','.join(arg_names)});", None)


def gen_symbol_creation(symbol_class, result_name):
    """
    Generate code to create a symbol of the given class from a result

    Example:
    ```
    var ark_result_symbol = new shared SymEntry(ark_result);
    ```

    Returns (chapel code, symbol name)
    """
    return (
        f"\tvar ark_result_symbol = new shared {symbol_class}({result_name});\n",
        "ark_result_symbol",
    )


def gen_response(result=None, is_symbol=False):
    """
    Generate code to return a response object

    Examples:
    ```
    return st.insert(ark_result);
    return MsgTuple.fromScalar(x);
    return MsgTuple.success();
    ```
    """
    # TODO: need to handle tuple and list return types differently here
    if result:
        if is_symbol:
            return f"\treturn {SYMTAB_FORMAL_NAME}.insert({result});"
        else:
            return f"\treturn {RESPONSE_TYPE_NAME}.fromScalar({result});"
    else:
        return f"\treturn {RESPONSE_TYPE_NAME}.success();"


def gen_command_proc(name, return_type, formals, mod_name):
    """
    Generate a chapel command procedure that calls a user-defined procedure

    * name: the name of the user-defined procedure
    * return_type: the return type of the user-defined procedure
    * formals: a list of tuples in the format (name, storage kind, type expression)
        representing the formal parameters of the user-defined procedure

    Returns a tuple of:
        * the chapel code for the command procedure
        * the name of the command procedure
        * a boolean indicating whether the command has generic (param/type) formals
        * a list of tuples in the format (name, storage kind, type expression)
            representing the generic formals of the command procedure

    proc <cmd_name>(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, <param/type args...>): MsgTuple throws {
        <unpack arguments>(<param/type args...>)
        <call user function>
        <store symbols in the symbol table> (if necessary)
        <return response object>
    }

    """

    arg_unpack, command_formals = gen_arg_unpacking(formals)
    is_generic_command = len(command_formals) > 0
    signature, cmd_name = gen_signature(name, command_formals)
    fn_call, result_name = gen_user_function_call(
        name, [f.name for f in formals], mod_name, return_type
    )

    # get the names of the array-elt-type queries in the formals
    array_etype_queries = [
        f.info[1].name
        for f in formals
        if (
            f.kind == formalKind.ARRAY
            and len(f.info) > 0
            and isinstance(f.info[1], FormalQuery)
        )
    ]

    print(array_etype_queries)

    # assume the returned type is a symbol if it's an identifier that is not a scalar or type-query reference
    # or if it is a `SymEntry` type-constructor call
    returns_symbol = (
        return_type
        and (
            isinstance(return_type, chapel.Identifier)
            and return_type.name() not in chapel_scalar_types
            and return_type.name() not in array_etype_queries
        )
        or (
            # TODO: generalize this to any class type identifier or class type-constructor call that
            # inherits from 'AbstractSymEntry'
            isinstance(return_type, chapel.FnCall)
            and return_type.called_expression().name() == ARRAY_ENTRY_CLASS_NAME
        )
    )
    returns_array = (
        return_type
        and isinstance(return_type, chapel.BracketLoop)
        and return_type.is_maybe_array_type()
    )

    if returns_array:
        symbol_creation, result_name = gen_symbol_creation(
            ARRAY_ENTRY_CLASS_NAME, result_name
        )
    else:
        symbol_creation = ""

    response = gen_response(result_name, returns_symbol or returns_array)

    command_proc = "\n".join(
        [signature, arg_unpack, fn_call, symbol_creation, response, "}"]
    )

    return (command_proc, cmd_name, is_generic_command, command_formals)


def stamp_out_command(
    config, formals, name, cmd_prefix, mod_name, line_num, is_user_proc
):
    """
    Yield instantiations of a generic command with using the
    values from the configuration file

    The instantiated commands will be registered in the command table

    Arguments:
    * config: the configuration file
    * formals: a list of tuples in the format (name, storage kind, type expression)
        representing the generic formals of the command procedure
    * name: the name of the generic command procedure
    * cmd_prefix: the prefix to use for the command names
    * mod_name: the name of the module containing the command procedure
        (or the user-defined procedure that the command calls)

    The name of the instantiated command will be in the format: 'cmd_prefix<v1, v2, ...>'
    where v1, v2, ... are the values of the generic formals
    """
    formal_perms = generic_permutations(config, formals)

    for fp in formal_perms:
        stamp = stamp_generic_command(
            name, cmd_prefix, mod_name, fp, line_num, is_user_proc
        )
        yield stamp


def register_commands(config, source_files):
    """
    Create a chapel source file that registers all the procs annotated with the
    'arkouda.registerCommand' or 'arkouda.instantiateAndRegister' attributes
    """
    stamps = [
        "module Commands {",
        "use CommandMap, Message, MultiTypeSymbolTable, MultiTypeSymEntry;",
        "use BigInteger;",
        watermarkConfig(config),
    ]

    count = 0

    for filename, ctx in chapel.files_with_contexts(source_files):
        file_stamps = []
        found_annotation = False

        root, _ = next(chapel.each_matching(ctx.parse(filename), chapel.Module))
        mod_name = filename.split("/")[-1].split(".")[0]

        file_stamps.append(f"import {mod_name};")

        # register procs annotated with 'registerCommand',
        # creating generic commands and instantiations if necessary
        for fn, attr_call in annotated_procs(root, registerAttr):
            found_annotation = True

            name = fn.name()
            line_num = fn.location().start()[0]

            try:
                con_formals, gen_formals = get_formals(fn, True)
            except ValueError as e:
                error_message(f"registering '{name}'", e, fn.location())
                continue

            if prefix := attr_call["name"]:
                command_prefix = prefix.value()
            else:
                command_prefix = name

            if len(gen_formals) > 0:
                error_message(
                    f"registering '{name}'",
                    "generic formals are not allowed in commands",
                    fn.location(),
                )
                continue

            (cmd_proc, cmd_name, is_generic_cmd, cmd_gen_formals) = gen_command_proc(
                name, fn.return_type(), con_formals, mod_name
            )

            file_stamps.append(cmd_proc)
            count += 1

            if is_generic_cmd > 0:
                try:
                    for stamp in stamp_out_command(
                        config,
                        cmd_gen_formals,
                        cmd_name,
                        command_prefix,
                        mod_name,
                        line_num,
                        False,
                    ):
                        file_stamps.append(stamp)
                except ValueError as e:
                    error_message(f"registering '{name}'", e, fn.location())
                    continue
            else:
                file_stamps.append(
                    f"registerFunction('{command_prefix}', {cmd_name}, '{mod_name}', {line_num});"
                )

        # instantiate and register procs annotated with 'instantiateAndRegister'
        for fn, attr_call in annotated_procs(root, instAndRegisterAttr):
            found_annotation = True

            name = fn.name()
            line_num = fn.location().start()[0]

            try:
                con_formals, gen_formals = get_formals(fn, False)
            except ValueError as e:
                error_message(f"registering '{name}'", e, fn.location())
                continue

            if not valid_generic_command_signature(fn, con_formals):
                error_message(
                    f"registering '{name}'",
                    "generic instantiation of commands must have the signature "
                    + "'proc <name>(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws'",
                    fn.location(),
                )
                continue

            if prefix := attr_call["prefix"]:
                command_prefix = prefix.value()
            else:
                command_prefix = name

            try:
                for stamp in stamp_out_command(
                    config, gen_formals, name, command_prefix, mod_name, line_num, True
                ):
                    file_stamps.append(stamp)
                    count += 1
            except ValueError as e:
                error_message(f"registering '{name}'", e, fn.location())
                continue

        if found_annotation:
            stamps.extend(file_stamps)

    stamps.append("}")

    return ("\n\n".join(stamps) + "\n", count)


def getModuleFiles(config, src_dir):
    with open(config, "r") as cfg_file:
        mods = []
        for line in itertools.chain(cfg_file.readlines(), DEFAULT_MODS):
            mod = line.split("#")[0].strip()
            if mod != "":
                mods.append(f"{mod}.chpl" if mod[0] == "/" else f"{src_dir}/{mod}.chpl")
        return mods


def watermarkConfig(config):
    return 'param regConfig = """\n' + json.dumps(config, indent=2) + '\n""";'


def main():
    config = json.load(open(sys.argv[1]))
    source_files = getModuleFiles(sys.argv[2], sys.argv[3])
    (chpl_src, n) = register_commands(config, source_files)

    with open(sys.argv[3] + "/registry/Commands.chpl", "w") as f:
        f.write(chpl_src.replace("\t", "  "))

    print("registered ", n, " commands from ", len(source_files), " modules")


if __name__ == "__main__":
    main()
