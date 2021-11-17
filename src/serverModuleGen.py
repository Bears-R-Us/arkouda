def GetModules(filename):
    with open(filename) as configfile:
        modules = configfile.readlines()
        ret = []
        for module in modules:
            module = (module.rstrip()).lstrip()
            if len(module) > 0 and module[0] != '#':
                ret.append(module)
        return ret

def GenerateServerIncludes(config_filename, reg_filename):
    serverfile = open(reg_filename, "w")
    serverfile.write("proc doRegister() {\n")
    for mod in GetModules(config_filename):
        serverfile.write("  import " + mod + ";\n")
        serverfile.write("  " + mod + ".registerMe();\n")
    serverfile.write("}\n")

GenerateServerIncludes("ServerModules.cfg", "src/ServerRegistration.chpl")
