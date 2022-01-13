def getModules(filename):
    with open(filename) as configfile:
        modules = configfile.readlines()
        ret = []
        for module in modules:
            module = (module.rstrip()).lstrip()
            if len(module) > 0 and module[0] != '#':
                ret.append(module)
        return ret

def generateServerIncludes(config_filename, reg_filename):
    serverfile = open(reg_filename, "w")
    serverfile.write("proc doRegister() {\n")
    for mod in getModules(config_filename):
        serverfile.write(f"  import {mod};\n  {mod}.registerMe();\n")
    serverfile.write("}\n")

if __name__ == "__main__":
    import sys
    generateServerIncludes(sys.argv[1], "src/ServerRegistration.chpl")
