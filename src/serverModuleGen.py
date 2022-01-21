import os, sys

def getModules(filename):
    with open(filename) as configfile:
        modules = configfile.readlines()
        ret = []
        for module in modules:
            module = (module.rstrip()).lstrip()
            if len(module) > 0 and module[0] != '#':
                if module[0] == '/':
                    ret.append(module[module.rfind('/')+1:])
                else:
                    ret.append(module)
        return ret

def generateServerIncludes(config_filename, reg_filename):
    serverfile = open(reg_filename, "w")
    serverfile.write("proc doRegister() {\n")
    for mod in getModules(config_filename):
        if mod.strip() == "ParquetMsg" and "ARKOUDA_SERVER_PARQUET_SUPPORT" not in os.environ:
            print("**WARNING**: ParquetMsg module declared in ServerModules.cfg but ARKOUDA_SERVER_PARQUET_SUPPORT is not set.", file=sys.stderr)
            print("**WARNING**: ParquetMsg module will NOT be built.", file=sys.stderr)
        else:
            serverfile.write(f"  import {mod};\n  {mod}.registerMe();\n")

    serverfile.write("}\n")

def generateSearchPaths(filename):
    with open(filename) as configfile:
        modules = configfile.readlines()
        ret = ""
        for module in modules:
            module = (module.rstrip()).lstrip()
            # Represents a path to a file
            if len(module) > 0 and module[0] == '/':
                ret += f" {module}.chpl"
                
        return ret
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        generateServerIncludes(sys.argv[1], "src/ServerRegistration.chpl")
    else:
        print(generateSearchPaths(sys.argv[1]))
