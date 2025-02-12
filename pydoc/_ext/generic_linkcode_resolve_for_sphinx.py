# the sphinx extension 'sphinx.ext.viewcode' links documentation to an online
# code repository but requires to bind the code to the url through a user
# specific `linkcode_resolve` function. This implementation should be fairly
# generic and easily adaptable.
#
# License: Public Domain, CC0 1.0 Universal (CC0 1.0)
# From: https://gist.github.com/nlgranger/55ff2e7ff10c280731348a16d569cb73

import inspect
import os
import subprocess
import sys

linkcode_revision = "master"
try:
    # lock to commit number
    cmd = "git log -n1 --pretty=%H"
    head = subprocess.check_output(cmd.split()).strip().decode("utf-8")
    linkcode_revision = head

    # if we are on master's HEAD, use master as reference
    cmd = "git log --first-parent master -n1 --pretty=%H"
    master = subprocess.check_output(cmd.split()).strip().decode("utf-8")
    if head == master:
        linkcode_revision = "master"

    # if we have a tag, use tag as reference
    cmd = "git describe --exact-match --tags " + head
    tag = subprocess.check_output(cmd.split(" ")).strip().decode("utf-8")
    linkcode_revision = tag

except subprocess.CalledProcessError:
    pass

linkcode_url = (
    "https://github.com/nlgranger/SeqTools/blob/"
    + linkcode_revision
    + "/{filepath}#L{linestart}-L{linestop}"
)


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None

    modname = info["module"]
    topmodulename = modname.split(".")[0]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        modpath = pkg_resources.require(topmodulename)[0].location
        filepath = os.path.relpath(inspect.getsourcefile(obj), modpath)
        if filepath is None:
            return
    except Exception:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    return linkcode_url.format(filepath=filepath, linestart=linestart, linestop=linestop)
