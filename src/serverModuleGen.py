import os
import sys
import re


def getModules(filename):
    with open(filename) as configfile:
        modules = configfile.readlines()
        ret = []
        for module in modules:
            module = module.split("#")[0].strip()
            if module:
                ret.append(module)
        return ret

def getMaxDims(filename):
    with open(filename, 'r') as configfile:
        return int(re.search(r'max_array_dims: (\d*)', configfile.read()).group(1))

def ndStamp(msg_proc_name, base_proc_name, command_name, d):
    return f"""
    proc {msg_proc_name}{d}D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return {base_proc_name}(cmd, msgArgs, st, {d});

    registerFunction("{command_name}{d}D", {msg_proc_name}{d}D);
    """

def ndMultiStamp(msg_proc_name, base_proc_name, command_name, d1, d2):
    return f"""
    proc {msg_proc_name}{d1}Dx{d2}D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws
        do return {base_proc_name}(cmd, msgArgs, st, {d1}, {d2});

    registerFunction("{command_name}{d1}Dx{d2}D", {msg_proc_name}{d1}Dx{d2}D);
    """

def ndStampBinary(msg_proc_name, base_proc_name, command_name, d):
    return f"""
    proc {msg_proc_name}{d}D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): bytes throws
        do return {base_proc_name}(cmd, msgArgs, st, {d});

    registerBinaryFunction("{command_name}{d}D", {msg_proc_name}{d}D);
    """

# TODO: use Chapel's compiler bindings to do this more robustly
def stampOutNDArrayHandlers(mod, src_dir, stamp_file, max_dims):
    with open(f"{src_dir}/{mod}.chpl", 'r') as src_file:
        found_annotation = False
        text = src_file.read()

        # find each procedure annotated with '@arkouda.registerND'
        #  (with an optional 'cmd_prefix' argument)
        #            group 0  \/                  1 \/            2 \/                          3 \/
        for m in re.finditer(r'\@arkouda\.registerND(\(cmd_prefix=\"([a-zA-Z0-9]*)\"\))?\s*proc\s*([a-zA-Z0-9]*)\(', text):
            found_annotation = True
            g = m.groups()

            base_proc_name = g[2]
            msg_proc_name = "_nd_gen_" + base_proc_name

            if g[0] == None:
                # no 'cmd_prefix' argument
                command_name = base_proc_name.replace('Msg', '')
            else:
                # group 2 contains the 'cmd_prefix' argument
                command_name = g[1]

            # instantiate the message handler for each rank from 1..max_dims
            # and register the instantiated proc with a unique command name
            for d in range(1, max_dims+1):
                stamp_file.write(ndStamp(msg_proc_name, base_proc_name, command_name, d))

        # find each procedure annotated with '@arkouda.registerNDBinary'
        for m in re.finditer(r'\@arkouda\.registerNDBinary(\(cmd_prefix=\"([a-zA-Z0-9]*)\"\))?\s*proc\s*([a-zA-Z0-9]*)\(', text):
            found_annotation = True
            g = m.groups()

            base_proc_name = g[2]
            msg_proc_name = "_nd_gen_" + base_proc_name

            if g[0] == None:
                # no 'cmd_prefix' argument
                command_name = base_proc_name.replace('Msg', '')
            else:
                # group 2 contains the 'cmd_prefix' argument
                command_name = g[1]

            # instantiate the message handler for each rank from 1..max_dims
            # and register the instantiated proc with a unique command name
            for d in range(1, max_dims+1):
                stamp_file.write(ndStampBinary(msg_proc_name, base_proc_name, command_name, d))

        # include the source module in the stamp file if any procs were stamped out
        if found_annotation:
            stamp_file.write(f"use {mod};\n")

def specialBroadcastStamp(src_dir, stamp_file, max_dims):
    # broadcast is a special case because it has a different signature
    # including two param fields for the source and destination ranks
    # so we need to stamp out a separate handler for each rank pair
    # (e.g., broadcast1Dx1D, broadcast1Dx2D, broadcast1Dx3D, broadcast2Dx2D, broadcast2Dx3D etc.)
    for d1 in range(1, max_dims+1):
        for d2 in range(d1, max_dims+1):
            stamp_file.write(ndMultiStamp("_nd_gen_broadcast", "broadcastNDArray", "broadcast", d1, d2))

def generateServerIncludes(config_filename, src_dir):
    res = ""
    stamp_file_path = f"{src_dir}/multi_dim_support/nd_array_stamps.chpl"
    max_dims = getMaxDims(config_filename)

    with open (stamp_file_path, 'w') as stamp_file:
        stamp_file.write("use CommandMap, Message, MultiTypeSymbolTable;\n")

        for mod in getModules(config_filename):
            if mod[0] != '/':
                res += f" {src_dir}/{mod}.chpl"
            else:
                res += f" {mod}.chpl"

            stampOutNDArrayHandlers(mod, src_dir, stamp_file, max_dims)

        # explicitly stamp out basic message handlers
        stampOutNDArrayHandlers("MsgProcessing", src_dir, stamp_file, max_dims)
        stampOutNDArrayHandlers("GenSymIO", src_dir, stamp_file, max_dims)

        # handle broadcast specially
        specialBroadcastStamp(src_dir, stamp_file, max_dims)

    res += " " + stamp_file_path
    print(res)


if __name__ == "__main__":
    generateServerIncludes(sys.argv[1], sys.argv[2])
