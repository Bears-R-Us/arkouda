import optparse

"""
    This script will take a commandline argument that is the value to be encoded.
    Currently, we are only able to encode idna (returning the ascii string instead of bytes).
"""
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-v", "--value", dest="value", help="The value to be encoded.")
    parser.add_option("-f", "--file", dest="filename", help="The name of the file to write output to.")
    (options, args) = parser.parse_args()
    try:
        with open(options.filename, "w") as f:
            f.write(options.value.encode("idna").decode("ascii"))
    except Exception:
        with open(options.filename, "w") as f:
            f.write("")
