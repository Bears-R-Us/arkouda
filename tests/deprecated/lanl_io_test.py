#!/usr/bin/env python3

import sys, time, argparse
import arkouda as ak

if __name__ == '__main__':
    ak.v = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default=None)
    parser.add_argument('--port', default=None)
    parser.add_argument('hdffiles', nargs='+')

    args = parser.parse_args()

    ak.set_defaults()
    ak.v = False
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

    fields = ['srcIP', 'dstIP', 'srcPort', 'dstPort', 'start']
    
    nfDF = {field: ak.read_hdf(field, args.hdffiles) for field in fields}
    
    print(nfDF['start'])
    print(nfDF['srcIP'])
    print(nfDF['dstIP'])
    print(nfDF['srcPort'])
    print(nfDF['dstPort'])
    print(nfDF)
    
    print(ak.info(ak.AllSymbols))

    print("mem used: ", ak.get_mem_used())
    
    u,c = ak.unique(nfDF['srcIP'],return_counts=True)
    print("unique values = ", u.size,u)
    print("value counts = ", c.size,c)
    
    u,c = ak.unique(nfDF['dstIP'],return_counts=True)
    print("unique values = ", u.size,u)
    print("value counts = ", c.size,c)
    
    u,c = ak.unique(nfDF['srcPort'],return_counts=True)
    print("unique values = ", u.size,u)
    print("value counts = ", c.size,c)
    
    u,c = ak.unique(nfDF['dstPort'],return_counts=True)
    print("unique values = ", u.size,u)
    print("value counts = ", c.size,c)
    
    ak.shutdown()
    #ak.disconnect()
    
