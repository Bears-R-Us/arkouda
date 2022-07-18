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


def generateServerIncludes(config_filename, src_dir):
    res = ""
    for mod in getModules(config_filename):
        res += f" {src_dir}/{mod}.chpl"

    print(res)


if __name__ == "__main__":
    generateServerIncludes(sys.argv[1], sys.argv[2])
