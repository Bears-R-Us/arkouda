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

# TODO: use Chapel's compiler bindings to do this in a more principled manner
def stampOutNDArrayHandlers(mod, src_dir, stamp_file, max_dims):
    with open(f"{src_dir}/{mod}.chpl", 'r') as src_file:
        found_annotation = False

        # find each procedure annotated with '@arkouda.registerND'
        for m in re.finditer(r'\@arkouda\.registerND\s*proc\s*([a-zA-Z0-9]*)\(', src_file.read()):
            found_annotation = True
            msg_proc_name = m.group(1)
            command_name = msg_proc_name.replace('Msg', '')

            # instantiate the message handler for each rank up to max_dims
            # and register the instantiated proc with a unique command name
            for d in range(1, max_dims+1):
                stamp_file.write(f"""
                proc {msg_proc_name}{d}D(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws do
                    return {msg_proc_name}(cmd, msgArgs, st, {d});

                registerFunction("{command_name}{d}D", {msg_proc_name}{d}D);
                """)

        # include the source module in the stamp file
        if found_annotation:
            stamp_file.write(f"use {mod};\n")

def generateServerIncludes(config_filename, src_dir, max_dims):
    res = ""
    stamp_file_path = f"{src_dir}/multi_dim_support/nd_array_stamps.chpl"
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

    res += " " + stamp_file_path
    print(res)


if __name__ == "__main__":
    generateServerIncludes(sys.argv[1], sys.argv[2], int(sys.argv[3]))
