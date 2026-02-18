import sys


def getModules(config):
    with open(config, "r") as cfg_file:
        mods = []
        for line in cfg_file.readlines():
            mod = line.split("#")[0].strip()
            if mod != "":
                mods.append(mod)
        return mods


def getModuleFiles(mods, src_dir):
    return " ".join([f"{mod}.chpl" if mod[0] == "/" else f"{src_dir}/{mod}.chpl" for mod in mods])


def parseServerConfig(config_filename, src_dir):
    # Create a list of module source files to include in the server build commands
    modules = getModules(config_filename)
    module_source_files = getModuleFiles(modules, src_dir)

    print(f"{module_source_files}")


if __name__ == "__main__":
    parseServerConfig(sys.argv[1], sys.argv[2])
