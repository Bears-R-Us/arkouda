import os
import sys
import re
import json

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

def ndStamp(nd_msg_handler_name, command_name, d):
    msg_proc_name = f"arkouda_nd_stamp_{nd_msg_handler_name}{d}D"
    return \
    f"proc {msg_proc_name}(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws\n" + \
    f"    do return {nd_msg_handler_name}(cmd, msgArgs, st, {d});\n" + \
    f"registerFunction(\"{command_name}{d}D\", {msg_proc_name});\n\n"

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

def createNDHandlerInstantiations(config, modules, src_dir):
    max_dims = config["max_array_dims"]
    filename = f"{src_dir}/nd_support/nd_array_stamps.chpl"

    with open(filename, 'w') as stamps:
        stamps.write("use CommandMap, Message, MultiTypeSymbolTable;\n\n")

        # stamp out message handlers for each included module with
        # '@arkouda.registerND' annotations
        for mod in modules:
            stampOutModule(mod, src_dir, stamps, max_dims)

        # explicitly stamp out basic message handlers
        stampOutModule("MsgProcessing", src_dir, stamps, max_dims)

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
