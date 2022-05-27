import os
import sys


def getModules(filename):
    with open(filename) as configfile:
        modules = configfile.readlines()
        ret = []
        for module in modules:
            module = module.split("#")[0].split("/")[-1].strip()
            if module:
                ret.append(module)
        return ret


def generateServerIncludes(config_filename, reg_filename):
    serverfile = open(reg_filename, "w")
    serverfile.write("proc doRegister() {\n")
    for mod in getModules(config_filename):
        serverfile.write(f"  import {mod};\n  {mod}.registerMe();\n")

    serverfile.write("}\n")


if __name__ == "__main__":
    generateServerIncludes(sys.argv[1], "src/ServerRegistration.chpl")
