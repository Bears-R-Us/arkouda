import codecs
import optparse

"""
    This script will take a commandline argument that is the value to be decoded.
    Currently, we are only able to decode idna strings. Bytes are not functional..
"""
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-v", "--value", dest="value", help="The value to be encoded.")
    parser.add_option("-f", "--file", dest="filename", help="The name of the file to write output to.")
    (options, args) = parser.parse_args()
    try:
        decoded = options.value.encode("ascii").decode("idna")
        if decoded.strip()[:4] == "xn--":
            raise ValueError("Invalid Encoding")
        with codecs.open(options.filename, "w", "utf-8") as f:
            f.write(decoded)
    except Exception:
        with open(options.filename, "w") as f:
            f.write("")
