import chapel
import sys
import json

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


def get_formals(fn):
    """
    Get a function's formal parameters, separating them into concrete and
    generic (param and type) formals
    The (name, storage kind, and type expression) of each formal is returned
    """

    def info_tuple(formal):
        if te := formal.type_expression():
            if isinstance(te, chapel.BracketLoop):
                if te.is_maybe_array_type():
                    ten = "<array>"
                else:
                    raise ValueError("invalid formal type expression")
            elif isinstance(te, chapel.FnCall):
                if ce := te.called_expression():
                    if isinstance(ce, chapel.Identifier):
                        call_name = ce.name()
                        if call_name == "list":
                            list_type = list(te.actuals())[0].name()
                            if list_type not in chapel_scalar_types:
                                error_message(
                                    f"registering '{fn.name()}'",
                                    f"unsupported formal type for registration {list_type}",
                                )
                                ten = None
                            else:
                                ten = f"list,{list_type}"
                        elif call_name == "borrowed":
                            ten = call_name + " " + list(te.actuals())[0].name()
                        else:
                            error_message(
                                f"registering '{fn.name()}'",
                                f"unsupported composite type expression for registration {call_name}",
                            )
                            ten = None
                    else:
                        ten = None
                else:
                    ten = None
            else:
                ten = te.name()
        else:
            ten = None
        return (formal.name(), formal.storage_kind(), ten)

    con_formals = []
    gen_formals = []
    for formal in fn.formals():
        if isinstance(formal, (chapel.TupleDecl, chapel.VarArgFormal)):
            raise ValueError(
                "registration of procedures with tuple or vararg formals is not yet supported."
            )
        elif isinstance(formal, (chapel.Formal, chapel.AnonFormal)):
            if formal.storage_kind() in ["type", "param"]:
                gen_formals.append(info_tuple(formal))
            else:
                con_formals.append(info_tuple(formal))
    return con_formals, gen_formals


def stamp_generic_command(generic_proc_name, prefix, module_name, formals, line_num, is_user_proc):
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

    stamp_name = f"ark_{prefix}_" + "_".join(
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
    for formal_name, storage_kind, _ in gen_formals:
        name_components = formal_name.split("_")

        if len(name_components) < 2:
            raise ValueError(
                f"invalid generic formal '{formal_name}; "
                + "generic formals must be in the format '<param-class>_<param-name>[_...]' "
                + "to be instantiated with values from the configuration file"
            )

        pclass = name_components[0]  # ex: 'array'
        pname = name_components[1]  # ex: 'nd'

        if pclass not in config["parameter_classes"].keys():
            raise ValueError(
                f"generic formal '{formal_name}' is not a valid parameter class; "
                + "please check the 'parameter_classes' field in the configuration file"
            )

        if pname not in config["parameter_classes"][pclass].keys():
            raise ValueError(
                f"parameter class '{pclass}' has no parameter '{pname}'; "
                + "please check the 'parameter_classes' field in the configuration file"
            )

        to_permute[formal_name] = parse_param_class_value(
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
            con_formals[0][0] != "cmd"
            or con_formals[0][2] != "string"
            or con_formals[1][0] != ARGS_FORMAL_NAME
            or con_formals[1][1] != ARGS_FORMAL_INTENT
            or con_formals[1][2] != ARGS_FORMAL_TYPE
            or con_formals[2][0] != SYMTAB_FORMAL_NAME
            or con_formals[2][1] != SYMTAB_FORMAL_INTENT
            or con_formals[2][2] != SYMTAB_FORMAL_TYPE
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
def unpack_array_arg(arg_name, array_count):
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
    dtype_arg_name = "array_dtype_" + str(array_count)
    nd_arg_name = "array_nd_" + str(array_count)
    return (
        f"\tvar {arg_name}_array_sym = {SYMTAB_FORMAL_NAME}[{ARGS_FORMAL_NAME}['{arg_name}']]: {ARRAY_ENTRY_CLASS_NAME}({dtype_arg_name}, {nd_arg_name});\n"
        + f"\tref {arg_name} = {arg_name}_array_sym.a;",
        [(dtype_arg_name, "type", None), (nd_arg_name, "param", "int")],
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


def unpack_tuple_arg(arg_name, tuple_size, scalar_type):
    """
    Generate the code to unpack a tuple argument

    Example:
    ```
    var x = msgArgs['x'].getTuple(int, 2);
    ```
    """
    return f"\tvar {arg_name} = {ARGS_FORMAL_NAME}['{arg_name}'].toScalarTuple({scalar_type}, {tuple_size});"


def unpack_list_arg(arg_name, arg_type):
    """
    Generate the code to unpack a list argument

    Example:
    ```
    var x = msgArgs['x'].toScalarList(int);
    ```
    """

    list_elt_type = arg_type.split(",")[1]
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
        arg_strings = [
            f"{kind} {name}: {ft}" if ft else f"{kind} {name}"
            for (name, kind, ft) in generic_args
        ]
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

    for fname, fintent, ftype in formals:
        if ftype in chapel_scalar_types:
            unpack_lines.append(unpack_scalar_arg(fname, ftype))
        elif ftype == "<array>":
            code, array_args = unpack_array_arg(fname, array_arg_counter)
            unpack_lines.append(code)
            generic_args += array_args
            array_arg_counter += 1
        elif "list" in ftype:
            unpack_lines.append(unpack_list_arg(fname, ftype))
        elif ftype == "tuple":
            print("ERROR: tuple and list types are not yet supported")
            continue
        else:
            # TODO: handle generic user-defined types
            unpack_lines.append(unpack_user_symbol(fname, ftype))

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
        return "\treturn {RESPONSE_TYPE_NAME}.success();"


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
        name, [f[0] for f in formals], mod_name, return_type
    )

    returns_symbol = (
        return_type
        and isinstance(return_type, chapel.Identifier)
        and return_type.name() not in chapel_scalar_types
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


def stamp_out_command(config, formals, name, cmd_prefix, mod_name, line_num, is_user_proc):
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
        stamp = stamp_generic_command(name, cmd_prefix, mod_name, fp, line_num, is_user_proc)
        yield stamp


def register_commands(config, source_files):
    """
    Create a chapel source file that registers all the procs annotated with the
    'arkouda.registerCommand' or 'arkouda.instantiateAndRegister' attributes
    """
    stamps = [
        "module Commands {",
        "use CommandMap, Message, MultiTypeSymbolTable, MultiTypeSymEntry;",
    ]

    count = 0

    for filename, ctx in chapel.files_with_contexts(source_files):
        root = ctx.parse(filename)[0]
        mod_name = filename.split("/")[-1].split(".")[0]

        stamps.append(f"import {mod_name};")

        # register procs annotated with 'registerCommand',
        # creating generic commands and instantiations if necessary
        for fn, attr_call in annotated_procs(root, registerAttr):
            name = fn.name()
            line_num = fn.location().start()[0]

            try:
                con_formals, gen_formals = get_formals(fn)
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
            stamps.append(cmd_proc)
            count += 1;

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
                        stamps.append(stamp)
                except ValueError as e:
                    error_message(f"registering '{name}'", e, fn.location())
                    continue
            else:
                stamps.append(
                    f"registerFunction('{command_prefix}', {cmd_name}, '{mod_name}', {line_num});"
                )

        # instantiate and register procs annotated with 'instantiateAndRegister'
        for fn, attr_call in annotated_procs(root, instAndRegisterAttr):
            name = fn.name()
            line_num = fn.location().start()[0]

            try:
                con_formals, gen_formals = get_formals(fn)
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
                    stamps.append(stamp)
                    count += 1
            except ValueError as e:
                error_message(f"registering '{name}'", e, fn.location())
                continue

    stamps.append("}")

    return ("\n\n".join(stamps), count)


def getModuleFiles(config, src_dir):
    with open(config, 'r') as cfg_file:
        mods = []
        for line in cfg_file.readlines():
            mod = line.split("#")[0].strip()
            if mod != "":
                mods.append(f"{mod}.chpl" if mod[0] == '/' else f"{src_dir}/{mod}.chpl")
        return mods


def main():
    config = json.load(open(sys.argv[1]))
    source_files = getModuleFiles(sys.argv[2], sys.argv[3])
    (chpl_src, n) = register_commands(config, source_files)

    with open(sys.argv[3] + "/registry/Commands.chpl", "w") as f:
        f.write(chpl_src.replace("\t", "  "))

    print("registered ", n, " commands from ", len(source_files), " modules")


if __name__ == "__main__":
    main()
