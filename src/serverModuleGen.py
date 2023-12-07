import os
import sys
import re
import json

def getModuleFiles(config, src_dir):
    return " ".join([f"{mod}.chpl" if mod[0] == '/' else f"{src_dir}/{mod}.chpl" \
                     for mod in config["modules"]])

def ndStamp(nd_msg_handler_name, command_name, d):
    msg_proc_name = f"_nd_stamp_{nd_msg_handler_name}{d}D"
    return f"""
    proc {msg_proc_name}(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return {nd_msg_handler_name}(cmd, msgArgs, st, {d});
    registerFunction("{command_name}{d}D", {msg_proc_name});
    """

def stampOutModule(mod, src_dir, stamp_file, max_dims):
    with open(f"{src_dir}/{mod}.chpl", 'r') as src_file:
        found_annotation = False

        # find each procedure annotated with '@arkouda.registerND'
        #  (with an optional 'cmd_prefix' argument)
        #            group 0  \/                  1 \/            2 \/                          3 \/
        for m in re.finditer(r'\@arkouda\.registerND(\(cmd_prefix=\"([a-zA-Z0-9]*)\"\))?\s*proc\s*([a-zA-Z0-9]*)\(', src_file.read()):
            found_annotation = True
            g = m.groups()
            proc_name = g[2]

            if g[0] == None:
                # no 'cmd_prefix' argument
                command_name = proc_name.replace('Msg', '')
            else:
                # group 2 contains the 'cmd_prefix' argument
                command_name = g[1]

            # instantiate the message handler for each rank from 1..max_dims
            # and register the instantiated proc with a unique command name
            for d in range(1, max_dims+1):
                stamp_file.write(ndStamp(proc_name, command_name, d))

        # include the source module in the stamp file if any procs were stamped out
        if found_annotation:
            stamp_file.write(f"use {mod};\n")

def createNDHandlerInstantiations(config, src_dir):
    max_dims = config["max_array_dims"]
    filename = f"{src_dir}/nd_support/nd_array_stamps.chpl"

    with open(filename, 'w') as stamps:
        stamps.write("use CommandMap, Message, MultiTypeSymbolTable;\n")

        # stamp out message handlers for each included module with
        # '@arkouda.registerND' annotations
        for mod in config["modules"]:
            stampOutModule(mod, src_dir, stamps, max_dims)

        # explicitly stamp out basic message handlers
        stampOutModule("MsgProcessing", src_dir, stamps, max_dims)

    return f"{filename} -sMaxArrayDims={max_dims}"

def getSupportedTypes(config):
    supportedFlags = []
    for t in ["uint8", "uint16", "uint32", "uint64", \
              "int8", "int16", "int32", "int64", \
              "float32", "float64", "complex64", "complex128", "bool"]:
        if config["supported_scalar_types"][t]:
            supportedFlags.append(f"-sSupports{t.capitalize()}=true")
    return " ".join(supportedFlags)

def parseServerConfig(config_filename, src_dir):
    config = json.load(open(config_filename))

    module_source_files = getModuleFiles(config, src_dir)
    nd_stamp_flags = createNDHandlerInstantiations(config, src_dir)
    type_flags = getSupportedTypes(config)

    print(f"{module_source_files} {nd_stamp_flags} {type_flags}")

if __name__ == "__main__":
    parseServerConfig(sys.argv[1], sys.argv[2])
