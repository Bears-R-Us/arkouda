import os
import sys


def getModules(filename):
    with open(filename) as configfile:
        modules = configfile.readlines()
        ret = []
        for module in modules:
            module = module.split("#")[0].strip()
            if module:
                ret.append(module)
        return ret


def generateServerIncludes(config_filename, src_dir):
    res = ""
    for mod in getModules(config_filename):
        if mod[0] != '/':
            res += f" {src_dir}/{mod}.chpl"
        else:
            res += f" {mod}.chpl"

    print(res)


if __name__ == "__main__":
    generateServerIncludes(sys.argv[1], sys.argv[2])
