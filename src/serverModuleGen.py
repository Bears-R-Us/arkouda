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
        if "/" in mod:
            serverfile.write(f"  require \"{mod}\";\n")
            mod = mod.split("/")[-1] # get only module name, not path
        serverfile.write(f"  import {mod};\n  {mod}.registerMe();\n")
    serverfile.write("}\n")

if __name__ == "__main__":
    import sys
    GenerateServerIncludes(sys.argv[1], "src/ServerRegistration.chpl")
