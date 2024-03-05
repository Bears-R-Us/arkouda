import os
import sys
import re
import json
import io
import math

# TODO: rewrite the nd-stamping portion of this script to use the compiler's frontend python bindings
# would remove the need for regex parsing, special handling of binary message handlers, problems
# with multiline proc signatures, etc.

def getModules(config):
    with open(config, 'r') as cfg_file:
        mods = []
        for line in cfg_file.readlines():
            mod = line.split("#")[0].strip()
            if mod != "":
                mods.append(mod)
        return mods

def getModuleFiles(mods, src_dir):
    return " ".join([f"{mod}.chpl" if mod[0] == '/' else f"{src_dir}/{mod}.chpl" \
                        for mod in mods])

def ndStamp(nd_msg_handler_name, cmd_prefix, d):
    msg_proc_name = f"arkouda_nd_stamp_{nd_msg_handler_name}{d}D"
    return \
    f"\nproc {msg_proc_name}(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws\n" + \
    f"    do return {nd_msg_handler_name}(cmd, msgArgs, st, {d});\n" + \
    f"registerFunction(\"{cmd_prefix}{d}D\", {msg_proc_name});\n"

def ndStampBinary(nd_msg_handler_name, cmd_prefix, d):
    msg_proc_name = f"arkouda_nd_stamp_{nd_msg_handler_name}{d}D"
    return \
    f"\nproc {msg_proc_name}(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): bytes throws\n" + \
    f"    do return {nd_msg_handler_name}(cmd, msgArgs, st, {d});\n" + \
    f"registerBinaryFunction(\"{cmd_prefix}{d}D\", {msg_proc_name});\n"

def ndStampArrayMsg(d):
    msg_proc_name = f"arkouda_nd_stamp_arrayMsg{d}D"
    return \
    f"\nproc {msg_proc_name}(cmd: string, msgArgs: borrowed MessageArgs, ref data: bytes, st: borrowed SymTab): MsgTuple throws\n" + \
    f"    do return arrayMsg(cmd, msgArgs, data, st, {d});\n" + \
    f"registerArrayFunction(\"array{d}D\", {msg_proc_name});\n"


def ndStampMultiRank(nd_msg_handler_name, cmd_prefix, d1, d2):
    msg_proc_name = f"arkouda_nd_stamp_{nd_msg_handler_name}{d1}Dx{d2}D"
    return \
    f"\nproc {msg_proc_name}(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws\n" + \
    f"    do return {nd_msg_handler_name}(cmd, msgArgs, st, {d1}, {d2});\n" + \
    f"registerFunction(\"{cmd_prefix}{d1}Dx{d2}D\", {msg_proc_name});\n"

def ndStampPermInc(nd_msg_handler_name, cmd_prefix, stamp_file, max_dims):
    # (e.g., broadcast1Dx1D, broadcast1Dx2D, broadcast1Dx3D, broadcast2Dx2D, broadcast2Dx3D etc.)
    for d1 in range(1, max_dims+1):
        for d2 in range(d1, max_dims+1):
            stamp_file.write(ndStampMultiRank(nd_msg_handler_name, cmd_prefix, d1, d2))

def ndStampPermDec(nd_msg_handler_name, cmd_prefix, stamp_file, max_dims):
    # (e.g., squeeze1Dx1D, squeeze2Dx1D, squeeze2Dx2D, squeeze3Dx1D, squeeze3Dx2D etc.)
    for d1 in range(1, max_dims+1):
        for d2 in range(1, d1+1):
            stamp_file.write(ndStampMultiRank(nd_msg_handler_name, cmd_prefix, d1, d2))

def ndStampPermAll(nd_msg_handler_name, cmd_prefix, stamp_file, max_dims):
    for d1 in range(1, max_dims+1):
        for d2 in range(1, max_dims+1):
            stamp_file.write(ndStampMultiRank(nd_msg_handler_name, cmd_prefix, d1, d2))

def stampOutModule(mod, src_dir, stamp_file, max_dims):
    with open(f"{src_dir}/{mod}.chpl", 'r') as src_file:
        found_annotation = False

        modOut = io.StringIO()
        modOut.write('/'*math.ceil((100-len(mod))/2) + mod + '/'*math.floor((100-len(mod))/2) + '\n')
        modOut.write(f"use {mod};")

        ftext = src_file.read()

        # find each procedure annotated with '@arkouda.registerND'
        #  (with an optional 'cmd_prefix' argument)
        #            group 0  \/                     1 \/          2 \/                               3 \/                        4\/
        for m in re.finditer(r'\@arkouda\.registerND\(?(cmd_prefix=\"([\[\]a-zA-Z0-9\-\=<>]*)\")?\)?\s*proc\s*([a-zA-Z0-9]*)\(.*\)\s*:\s*(bytes)?', ftext):
            found_annotation = True
            g = m.groups()
            proc_name = g[2]

            if g[0] == None:
                # no 'cmd_prefix' argument
                cmd_prefix = proc_name.replace('Msg', '')
            else:
                # group 2 contains the 'cmd_prefix' argument
                cmd_prefix = g[1]

            # if return type is bytes, this is a binary message handler
            binaryHandler = g[3] != None

            # instantiate the message handler for each rank from 1..max_dims
            # and register the instantiated proc with a unique command name
            for d in range(1, max_dims+1):
                if binaryHandler:
                    modOut.write(ndStampBinary(proc_name, cmd_prefix, d))
                else:
                    modOut.write(ndStamp(proc_name, cmd_prefix, d))

        # find each procedure annotated with '@arkouda.registerNDPerm[Inc|Dec|All]'
        #            group 0  \/                      1 \/            2 \/          3 \/                                4\/
        for m in re.finditer(r'\@arkouda\.registerNDPerm(Inc|Dec|All)\(?(cmd_prefix=\"([\[\]a-zA-Z0-9\-\=<>]*)\")?\)?\s*proc\s*([a-zA-Z0-9]*)', ftext):
            found_annotation = True
            g = m.groups()
            proc_name = g[3]

            if g[1] == None:
                cmd_prefix = proc_name.replace('Msg', '')
            else:
                cmd_prefix = g[2]

            if g[0] == "Inc":
                ndStampPermInc(proc_name, cmd_prefix, modOut, max_dims)
            elif g[0] == "Dec":
                ndStampPermDec(proc_name, cmd_prefix, modOut, max_dims)
            else:
                ndStampPermAll(proc_name, cmd_prefix, modOut, max_dims)

        # special handling for 'arrayMsg'
        for m in re.finditer(r'\@arkouda.registerArrayMsg', ftext):
            found_annotation = True

            for d in range(1, max_dims+1):
                modOut.write(ndStampArrayMsg(d))

        # include the source module in the stamp file if any procs were stamped out
        if found_annotation:
            modOut.write('/'*100 + '\n\n')
            stamp_file.write(modOut.getvalue())

def createNDHandlerInstantiations(config, modules, src_dir):
    max_dims = config["max_array_dims"]
    filename = f"{src_dir}/nd_support/nd_array_stamps.chpl"

    with open(filename, 'w') as stamps:
        stamps.write("// this file is generated by 'parseServerConfig.py'\n")
        stamps.write("use CommandMap, Message, MultiTypeSymbolTable;\n\n")

        # stamp out message handlers for each included module with
        # '@arkouda.registerND' annotations
        for mod in modules:
            stampOutModule(mod, src_dir, stamps, max_dims)

        # explicitly stamp out basic message handlers
        stampOutModule("MsgProcessing", src_dir, stamps, max_dims)
        stampOutModule("GenSymIO", src_dir, stamps, max_dims)

    return f"{filename} -sMaxArrayDims={max_dims}"

def getSupportedTypes(config):
    supportedFlags = []
    for t in ["uint8", "uint16", "uint32", "uint64", \
              "int8", "int16", "int32", "int64", \
              "float32", "float64", "complex64", "complex128", "bool"]:
        isSupported = "true" if config["supported_scalar_types"][t] else "false"
        supportedFlags.append(f"-sSupports{t.capitalize()}={isSupported}")
    return " ".join(supportedFlags)

def parseServerConfig(config_filename, src_dir):
    server_config = json.load(open('serverConfig.json', 'r'))

    # Create a list of module source files to include in the server build commands
    modules = getModules(config_filename)
    module_source_files = getModuleFiles(modules, src_dir)

    # Populate 'nd_array_stamps.chpl' with message handler instantiations
    # and produce relevant flags for building the server
    # All procedures in included modules annotated with '@arkouda.registerND'
    # will be instantiated for each rank from 1..max_array_dims
    nd_stamp_flags = createNDHandlerInstantiations(server_config, modules, src_dir)

    # Create build flags to designate which types the server should support
    type_flags = getSupportedTypes(server_config)

    print(f"{module_source_files} {nd_stamp_flags} {type_flags}")

if __name__ == "__main__":
    parseServerConfig(sys.argv[1], sys.argv[2])
