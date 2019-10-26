#!/usr/bin/env python3

import sys, time, argparse
import arkouda as ak

if __name__ == '__main__':
    ak.verbose = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default=None)
    parser.add_argument('--port', default=None)
    parser.add_argument('dsetName')
    parser.add_argument('filenames', nargs='+')

    args = parser.parse_args()
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
    print("Reading files...")
    start = time.time()
    a = ak.read_hdf(args.dsetName, args.filenames)
    end = time.time()
    t = end - start
    print(a)
    print(f'{t:.2f} seconds ({8*a.size/t:.2e} bytes/sec)')
    print("Testing bad filename...")
    badfilename = args.filenames[0] + '-should-not-exist-5473219431'
    try:
        ak.read_hdf(args.dsetName, args.filenames + [badfilename])
    except RuntimeError as e:
        print(e)
    print("Testing bad dsetName...")
    try:
        ak.read_hdf(args.dsetName+'-not-a-dset', args.filenames)
    except RuntimeError as e:
        print(e)
        
    ak.shutdown()
    sys.exit()
