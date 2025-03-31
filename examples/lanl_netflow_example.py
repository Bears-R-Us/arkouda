#!/usr/bin/env python3

import argparse
import sys
import time

import arkouda as ak

if __name__ == "__main__":
    ak.verbose = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("hdffiles", nargs="+")

    args = parser.parse_args()

    ak.set_defaults()
    ak.verbose = False
    if args.server is not None:
        if args.port is not None:
            ak.connect(server=args.server, port=args.port)
        else:
            ak.connect(server=args.server)
    else:
        if args.port is not None:
            ak.connect(port=args.port)
        else:
            ak.connect()

    print(ak.get_config())

    if len(args.hdffiles) == 0:
        print("usage: {} [--server server] [--port port] hdffiles ".format(sys.argv[0]))

    # fields in the files to read and create pdarrays in the dict
    fields = ["srcIP", "dstIP", "srcPort", "dstPort", "start"]

    # read in the files, all data from hdffiles
    # will be concatenated together in the fields/columns
    nfDF = {field: ak.read_hdf(field, args.hdffiles) for field in fields}

    # print out the pdarrays in the dict and their types
    print(nfDF["start"], nfDF["start"].dtype)
    print(nfDF["srcIP"], type(nfDF["srcIP"]))  # Strings dosen't have a dtype?!?
    print(nfDF["dstIP"], type(nfDF["dstIP"]))  # Strings dosen't have a dtype?!?
    print(nfDF["srcPort"], nfDF["srcPort"].dtype)
    print(nfDF["dstPort"], nfDF["dstPort"].dtype)
    print(nfDF)

    # print oput the symbols the server knows about
    print(ak.info(ak.AllSymbols))

    # print out how much memory is being used by the server
    print("mem used: ", ak.get_mem_used())

    # get the unique srcIP and the counts for each unique srcIP
    u, c = ak.unique(nfDF["srcIP"], return_counts=True)
    print("unique values = ", u.size, u)
    print("value counts = ", c.size, c)

    # get the unique dstIP and the counts for each unique dstIP
    u, c = ak.unique(nfDF["dstIP"], return_counts=True)
    print("unique values = ", u.size, u)
    print("value counts = ", c.size, c)

    # get the unique srcPort and the counts for each unique srcPort
    u, c = ak.unique(nfDF["srcPort"], return_counts=True)
    print("unique values = ", u.size, u)
    print("value counts = ", c.size, c)

    # get the unique dstPort and the counts for each unique dstPort
    u, c = ak.unique(nfDF["dstPort"], return_counts=True)
    print("unique values = ", u.size, u)
    print("value counts = ", c.size, c)

    # GroupBy (srcIP,dstIP)

    # GroupBy (srcIP,srcPort)

    # GruopBy (srcIP, srcPort, dstIP, dstPort)

    ak.shutdown()
    # ak.disconnect()
