# Automatic Function Registration Framework

Arkouda's server infrastructure is set up so that all commands have the following signature:

```chapel
proc someCommand(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws { ... }
```

Server commands are stored in a map that relates `command-name => command-procedure`

When a command is issued to the server, it is looked up in the map by name, and then executed with a `MessageArgs` object, representing the JSON arguments from the client, and a reference to the global symbol table (`st`) that allows the command to look up existing symbols or add new ones.

This framework (on its own) has some drawbacks:
* Writing argument parsing code is fairly rote.
* The code used to generate responses to the client is predictable and repetitive across commands.
* Most commands operate on arrays, and most of those commands do essentially the same operation for all array element types. They also tend to do the same thing regardless of the array's rank (dimensionality) — or at least there is usually a straightforward way to express the command in a rank-agnostic way. Because arrays and their operations need to be instantiated for a particular element type and rank, commands tend to include repeated code for each.

Arkouda's build system has a couple of annotations that make it easier to write commands without worrying about argument parsing, client responses, or re-writing code for multiple array types/ranks:

1. `registerCommand`: allows any Chapel procedure (with some restrictions) to be registered as a command — it does not need to adhere to the above signature. Argument parsing and response code are generated automatically.
2. `instantiateAndRegister`: makes it easier to instantiate a generic command for all combinations of array element type and rank specified in the server's configuration file.

The following sections will go over both annotations in more detail, explaining how they are used and what the current limitations are. This document finishes with a few technical details about how the annotations are implemented. For a deeper understanding of the functionality being automated by these annotations, it may be helpful to read the Messaging Overview tutorial first.

<a id="toc"></a>
## Table of Contents
* [The `registerCommand` annotation](#rc-annotation)
    * [requirements and notes](#rc-requirements)
* [The `instantiateAndRegister` annotation](#iar-annotation)
    * [parameter-classes and `registration-config.json`](#iar-param-classes)
      * [Note about using non-SymEntry classes w/ `registerCommand`](#non-sym-entry-classes)
      * [note about enum formals](#enum-formals)
    * [opting out of some instantiations](#iar-opt-out)
* [Build System Notes](#build-system)
    * [`make register-commands`](#bs-make)
    * [alternate config files](#bs-config)
    * [frontend bindings](#bs-bindings)

<a id="rc-annotation"></a>
## The `registerCommand` Annotation

The `@arkouda.registerCommand` annotation is designed to simplify the process of including a Chapel function as a command in an Arkouda server. Importantly, writing a command this way does not require an understanding of Arkouda's internals (i.e., the symbol table, the `MsgTuple` type, etc.).

Unlike typical Arkouda "message handlers", Chapel procedures annotated with `registerCommand` do not need to adhere to the standard function signature. Instead, they can have formal arguments and return values of any of the following types:

* Chapel scalars supported in Arkouda: `string`, `bool`, `int(?)`, `uint(?)`, `real(?)`, `bigint`
* Arrays of Chapel scalars
* Lists of Chapel scalars
* Homogenous tuples of Chapel scalars

For example, the following command creates a copy of an array and sets the element at the given index to the given value:

```chapel
@arkouda.registerCommand
proc foo(a: [?d] ?t, idx: d.rank*int, value: t): [] t {
    var b = a;
    b[idx] = value;
    return b;
}
```

Given a pdarray, `arr`, an index tuple that matches `arr`'s rank, and a value that matches `arr`'s dtype, the command is called as follows from the client:

```python
result = create_pdarray(
    generic_msg(
        cmd=f"foo<{arr.dtype},{arr.ndim}>",
        args={
            "a": arr,
            "idx": idx,
            "value": value
        },
    )
)
```

Note that the array argument's `dtype` and number of dimensions must be specified in the command. This is because the annotation tells Arkouda to create a separate instantiation of the procedure `foo` for each combination of `dtype` and rank supported by the server — each instantiation is registered as its own command.

<a id="rc-requirements"></a>
### requirements and notes

* all arguments must have type annotations
  - generic arguments are allowed as long as their type is specified (i.e., all three arguments in the above example are generic, but its clear that the first is an array with generic domain and element type, the second is a homogenous tuple whose size matches the array's rank, and the third is a scalar whose type matches the array's element type)
* the return type must be specified
  - this can be omitted if the procedure doesn't return anything
* procedures can throw
  - the server will propagate the error message to the client
* by default, the prefix in the command name will match the procedure's name
  - to overwrite this, use `@arkouda.registerCommand("alt-name")` or `@arkouda.registerCommand(name="alt-name")`
* if more than one array argument is specified, the command will be instantiated for all combinations of both array dtypes and ranks separately (e.g., if the server is configured to support 1D and 2D arrays, and 4 dtypes, then a procedure with two array arguments will be instantiated 64 (2 x 4 x 2 x 4) times).
  - if the array arguments are intended to always have the same rank, the same dtype, (or both), then `instantiateAndRegister` should be used instead
  - future work may allow something like the following to avoid instantiating all combinations: `proc foo(a: [?da] ?t, b: [?db] t) where da.rank == db.rank { ... }`
* if a particular command cannot support one or more of the types specified in the server's configuration, then `instantiateAndRegister` should be used instead.
  - there is also future work planned to remove this limitation

<a id="iar-annotation"></a>
## The `instantiateAndRegister` Annotation

The `@arkouda.instantiateAndRegister` annotation is a lower-level counterpart to `registerCommand`. It applies to Chapel procedures that adhere to the standard Arkouda command signature and provides support for instantiating additional generic arguments. This means that argument parsing and client response code are written manually, but there is more control over the procedure's behavior and instantiation.

Procedures registered with this annotation can include generic arguments (`type` or `param` arguments) after the typical argument list. These arguments typically correspond to array element types and ranks. For example, the same command as above could be written as follows using this annotation:

```chapel
@arkouda.instantiateAndRegister
proc foo(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws {
    const a = st[msgArgs["a"]]: SymEntry(array_dtype, array_nd),
          idx = msgArgs["idx"].toScalarTuple(int, array_nd),
          value = msgArgs["value"].toScalar(array_dtype);

    var b = a.a;
    b[idx] = value;

    return st.insert(new SymEntry(b));
}
```

<a id="iar-param-classes"></a>
### parameter-classes and `registration-config.json`

Notice that the names of the generic arguments both start with `array_`. This tells the annotation where in the `registration-config.json` file to find values to instantiate the procedure with. The relevant portion of the file might look like:

```json
"parameter_classes": {
    "array": {
        "nd": [1,2,3],
        "dtype": [
            "int",
            "real"
        ]
    }
}
```

The name `array_nd`, tells the build system to instantiate the argument with each of the values in the "nd" list under "array" (in this case: `1`, `2`, and `3`). Analogously, `array_dtype` will be instantiated with `int` and `real`. Overall, the procedure will be instantiated for all six combinations of those values.

If a procedure has more than one generic argument that refers to the same "parameter_classes" field, their names can be differentiated by appending a `_<string>` to the argument name. For example, here is a simplified version of the `cast` command that has two `type` arguments:

```chapel
@arkouda.instantiateAndRegister
proc cast(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
          type array_dtype_in,
          type array_dtype_out,
          param array_nd: int
): MsgTuple throws {
    const a = st[msgArgs["a"]]: SymEntry(array_dtype_in, array_nd);
    const b = a.a: array_dtype_out;
    return st.insert(new SymEntry(b));
}
```

Note, this procedure would be called like:
```python
result = create_pdarray(
    generic_msg(
        cmd=f"cast<{arr.dtype},{cast_to_dtype},{arr.ndim}>",
        args={"a": arr},
    )
)
```

Other parameter classes can also be added to the configuration file and used in a similar manner (both to instantiate array-related arguments, or for a completely different set of generic arguments). Note that `"array"` is a special parameter-class because the `registerCommand` annotation uses it to instantiate array arguments.

<a id=non-sym-entry-classes></a>
#### Note about using non-SymEntry classes w/ `registerCommand`:

In the Arkouda server, pdarray's are stored in the symbol-table using the `SymEntry` class. Because the vast majority of Arkouda commands operate on pdarrays, those annotated with `@arkouda.registerCommand` are given special syntactic treatment. The build system knows how to convert back and forth between a standard Chapel Array, and a `SymEntry`. In other words, the following signature:

```chapel
@arkouda.registerCommand
proc foo(ref x: [?d] ?t): [d] t { ... }
```

is essentially shorthand, so that command-implementors don't have to write a signature that involves the `SymEntry` class directly:

```chapel
@arkouda.registerCommand
proc foo(x: borrowed SymEntry(?)): SymEntry(x.etype, x.dimensions) { ... }
```

For non-SymEntry symbols (i.e., any other class that inherits from the base symbol-table class type: `AbstractSymEntry`), such a shorthand does not exist. For example, assume we have the following generic symbol-table class type:

```chapel
class MySym: AbstractSymEntry {
  type t;
  param n: int;
  ...
}
```

To create a command that accepts a generic `MySym`, the signature should look like this:

```chapel
@arkouda.registerCommand
proc bar(ms: borrowed MySym(?)): int { ... }
```

And a corresponding parameter class should be added to `registration-config.json`. It's name should match the name of the class type, and it should have one field for each generic field in the class:

```json
"parameter-classes": {
  ...
  "MySym": {
    "t": ["int", "bigint"],
    "n": [3, 4]
  }
}
```

At build-time, the generic fields in `bar`'s `MySym` argument will be instantiated for all permutations of the values in the configuration file (just as `SymEntry` is instantiated for the values in the `"array"` parameter-class). Each instantiation creates a unique command. In this example, those would be:

* `bar<Int64,3>`
* `bar<Int64,4>`
* `bar<Bigint,3>`
* `bar<Bigint,4>`


**Warning:** partially instantiated formal types (e.g., `proc bar(ms: borrowed MySym(int, ?)))`) are not yet supported by `registerCommand`. To achieve such behavior, `instantiateAndRegister` should be used instead:

```chapel
proc bar(md: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param MySym_n: int): MsgTuple throws {
  var ms = st[msgArgs['ms']]: borrowed MySym(int, MySym_n);
  ...
}
```

<a id="enum-formals"></a>
#### Note about `param` Enum formals:

In order to make a (group of) procedures generic over an enum, it's values can be specified in
the configuration file in the following format:

```json
"parameter-classes": {
  ...
  "group": {
    "field": {
      "__enum__": "ModuleName.EnumName",
      "__variants__": ["V1", "V2", "V3"]
    }
  }
  ...
}
```

The `__enum__` sub-field is used to add an import statement to `Commands.chpl`, so the module name where the enum is defined must be included.

The `__variants__` field specifies which variants of the enum the procedure should be instantiated for. For example, procedures with arguments in the form:

```chapel
param group_field[_...]: EnumName
```

will be instantiated by `instantiateAndRegister` with the following values: `EnumName.V1`, `EnumName.V2`, and `EnumName.V3`.

<a id="iar-opt-out"></a>
### opting out of some instantiations

Some commands cannot support all array element types or ranks specified in the configuration file. If this is the case, a where-clause can be used to avoid instantiation. For example, the following procedure opts out of supporting the `bigint` dtype by providing a separate overload that returns an error when `array_dtype == bigint`:

```chapel
use BigInteger;

...

@arkouda.instantiateAndRegister
proc bar(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws
  where array_dtype != bigint
{
  ...
}

proc bar(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, type array_dtype, param array_nd: int): MsgTuple throws
  where array_dtype == bigint
{
  return MsgTuple.error("bar does not support the 'bigint' dtype");
}
```

Note that only one of the two procedures should be annotated. In the future, the annotation itself may be modified to evaluate and respect where clauses (meaning that the second overload would not be necessary).

The same strategy can be used for `array_nd` arguments when a particular rank cannot be supported (if a procedure is only intended to support 1D arrays, it should simply not include an `array_nd` argument, and ranks should be hard coded to `1`);

<a id="build-system"></a>
## Build System Notes

<a id="bs-make"></a>
### `make register-commands`

As a part of the server's build process, the Makefile first runs `register-commands`, which generates `src/registry/Commands.chpl` via `src/registry/register_commands.py`.

The `Commands.chpl` file contains instantiations of all procedures (in the `ServerModules.cfg` file) annotated with `registerCommand` or `instantiateAndRegister`. This generated code is essentially what makes the annotations work.

<a id="bs-config"></a>
### alternate config files

Note that `make register commands` (and `make` itself) can also be run with different configuration files for modules and/or registration.

1. `ARKOUDA_CONFIG_FILE` can be set to any `.cfg` file with a list of modules. The build system will process any `@arkouda` annotations from those files.
2. `ARKOUDA_REGISTRATION_CONFIG` can be set to any `.json` file with the same format as `registration-config.json` (note: there must be an `"array"` parameter-class for `registerCommand` to handle array arguments).

<a id="bs-bindings"></a>
### the `chapel-py` frontend bindings

The `register_commands.py` script relies on the Chapel compiler's frontend Python bindings (aka `chapel-py`) to parse the annotated Chapel code. The `make register-commands` recipe attempts to invoke the script in a couple of different Python environments where `chapel-py` could be installed:

1. First, it attempts to use the local Chapel installation's virtual environment (`$CHPL_HOME/third-party/chpl-venv`)
    - this environment and `chapel-py` can be built by running `make chapel-py-venv` from $CHPL_HOME. If this command fails, it may help to `(cd $CHPL_HOME/third-party && make clobber)` first.
2. If this fails, due to a problem with Chapel's virtual environment, the script is run from the executing Python environment
    - although not recommended, one can set this up by manually building and installing `chapel-py` in their Arkouda development environment via `(cd $CHPL_HOME/chapel-py && pip install -e .)`
3. If that fails, the build system falls back to the `Commands.chpl` file included in the source tree. This works for default server builds, but won't include and new or modified commands, and won't respect any changes to `registration-config.json` or `ServerModules.cfg` (it is strictly a fallback for building a **default** server in an environment where chapel-py is not installed).
