def GetModules(filename):
    with open(filename) as configfile:
        modules = configfile.readlines()
        modules = [module.rstrip() for module in modules]
        return modules

def GenerateServerIncludes(config_filename, reg_filename):
    serverfile = open(reg_filename, "w")
    serverfile.write("proc doRegister() {\n")
    for mod in GetModules(config_filename):
        serverfile.write("  import " + mod + ";\n")
        serverfile.write("  " + mod + ".registerMe();\n")
    serverfile.write("}\n")

GenerateServerIncludes("ServerModuleConfig.txt", "src/ServerRegistration.chpl")
