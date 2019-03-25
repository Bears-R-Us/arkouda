#!/usr/bin/env python3

import sys, time, argparse
import arkouda as ak

if __name__ == '__main__':
    ak.v = False
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
    start = time.time()
    a = ak.read_hdf(args.dsetName, args.filenames)
    end = time.time()
    t = end - start
    print(a)
    print(f'{t:.2f} seconds ({8*a.size/t:.2e} bytes/sec)')
    ak.shutdown()
    sys.exit()
