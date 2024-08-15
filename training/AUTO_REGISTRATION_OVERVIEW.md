# Automatic Function Registration Framework

Arkouda's build system supports two annotations that simplify the process of creating new commands.

1. `registerCommand`: is intended to allow any Chapel procedure to be registered as a command without any use of Arkouda's internal infrastructure (symbolTable, message-args, etc.)
2. `instantiateAndRegister`: simplifies the process of creating commands that are generic over some fields (such as an array argument's rank or element type)

The following sections will go over each annotation in more detail, explaining how it is used and what its current limitations are. For a deeper understanding of the functionality being automated by these annotations, it may be helpful to read the Messaging Overview tutorial first. There is also some information about the build-system infrastructure that makes the annotations work


<a id="toc"></a>
## Table of Contents
* [The `registerCommand` annotation](#rc-annotation)
* [The `instantiateAndRegister` annotation](#iar-annotation)
    * [Limitations](#iar-limitations)
* [Build System Details](#build-system)
    * [Config Files](#config-file)

<a id="rc-annotation"></a>
## The `registerCommand` annotation

The `@arkouda.registerCommand` annotation is designed to simplify the process of including Chapel functions as commands in an Arkouda server. This applies to cases where there is some existing functionality written in Chapel and cases where new functionality is required and Chapel's performance is needed.

Unlike typical Arkouda "message handlers", Chapel procedures annotated with `registerCommand` do not need to adhere to the standard function signature:

```chapel
proc fooMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws [ ... ]
```

Instead, they can have formal arguments and return values of any of the following types:

* Chapel scalars supported in Arkouda: `string`, `bool`, `int(?)`, `uint(?)`, `real(?)`, `complex(?)`, `bigint`
* Arrays of Chapel scalars
* Lists of Chapel scalars
* Homogenous tuples of Chapel scalars

<a id="iar-annotation"></a>
## The `instantiateAndRegister` annotation

<a id="iar-limitations"></a>
### Limitations

<a id="build-system"></a>
## Build System Details

<a id="config-file"></a>
### Config Files
