import sys
import optparse

"""
    This script will take a commandline argument that is the value to be encoded.
    Currently, we are only able to encode idna (returning the ascii string instead of bytes).
"""
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-v", "--value", dest="value",
                      help="The value to be encoded.")
    (options, args) = parser.parse_args()
    try:
        sys.stdout.write(options.value.encode("idna").decode("ascii"))
    except Exception:
        sys.stdout.write("")
