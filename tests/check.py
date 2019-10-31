#!/usr/bin/env python3                                                         

import argparse
import os
import sys
import socket
import time

import numpy as np
import arkouda as ak


def create_parser():
    parser = argparse.ArgumentParser(description="Sanity check arkouda_server")
    parser.add_argument('--hostname', help='Hostname of arkouda server',
                        default='localhost')
    parser.add_argument('--port', type=int, help='Port of arkouda server',
                        default=5555)
    parser.add_argument('--server-connection-info',
                        help='File containing server `hostname:port`',
                        default=os.getenv('ARKOUDA_SERVER_CONNECTION_INFO'))
    parser.add_argument('--shutdown-server',
                        help='Shutdown arkouda server',
                        default=False, action='store_true')
    return parser

parser = create_parser()
args = parser.parse_args()

print(">>> Sanity checks on the arkouda_server")

if args.server_connection_info:
    while not os.path.exists(args.server_connection_info):
        time.sleep(1)
    with open(args.server_connection_info, 'r') as f:
        (args.hostname, args.port) = f.readline().split(':')
        args.port = int(args.port)
        if args.hostname == socket.gethostname():
            args.hostname = 'localhost'

ak.verbose = False
ak.connect(args.hostname, args.port)

N = 1_000_000


errors = False
def pass_fail(f):
    global errors
    errors = errors or not f
    return ("Passed" if f else "Failed")

def check_arange(N):
    # create np version
    a = np.arange(N)
    # create ak version
    b = ak.arange(N)
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())
    
print("check arange :", check_arange(N))

def check_linspace(N):
    # create np version
    a = np.linspace(10, 20, N)
    # create ak version
    b = ak.linspace(10, 20, N)
    # print(a,b)
    f = np.allclose(a, b.to_ndarray())
    return pass_fail(f)

print("check linspace :", check_linspace(N))

def check_ones(N):
    # create np version
    a = np.ones(N)
    # create ak version
    b = ak.ones(N)
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

print("check ones :", check_ones(N))

def check_zeros(N):
    # create np version
    a = np.zeros(N)
    # create ak version
    b = ak.zeros(N)
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

print("check zeros :", check_zeros(N))

def check_argsort(N):
    # create np version
    a = np.arange(N)
    a = a[::-1]
    iv = np.argsort(a)
    a = a[iv]
    # create ak version
    b = ak.arange(N)
    b = b[::-1]
    iv = ak.argsort(b)
    b = b[iv]
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

print("check argsort :", check_argsort(N))

def check_sort(N):
    # create np version
    a = np.arange(N)
    a = a[::-1]
    a = np.sort(a)
    # create ak version
    b = ak.arange(N)
    b = b[::-1]
    b = ak.sort(b)
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

print("check sort :", check_sort(N))

def check_get_slice(N):
    # create np version
    a = np.ones(N)
    a = a[::2]
    # create ak version
    b = ak.ones(N)
    b = b[::2]
    # print(a,b)
    c = a == b.to_ndarray()
    return pass_fail(c.all())

print("check get slice [::2] :", check_get_slice(N))

def check_set_slice_value(N):
    # create np version
    a = np.ones(N)
    a[::2] = -1
    # create ak version
    b = ak.ones(N)
    b[::2] = -1
    # print(a,b)
    c = a == b.to_ndarray()
    return pass_fail(c.all())

print("check set slice [::2] = value:", check_set_slice_value(N))

def check_set_slice(N):
    # create np version
    a = np.ones(N)
    a[::2] = a[::2] * -1
    # create ak version
    b = ak.ones(N)
    b[::2] = b[::2] * -1
    # print(a,b)
    c = a == b.to_ndarray()
    return pass_fail(c.all())

print("check set slice [::2] = pda:", check_set_slice(N))

def check_get_bool_iv(N):
    # create np version
    a = np.arange(N)
    a = a[a < N//2]
    # create ak version
    b = ak.arange(N)
    b = b[b < N//2]
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

print("check (compressing) get bool iv :", check_get_bool_iv(N))

def check_set_bool_iv_value(N):
    # create np version
    a = np.arange(N)
    a[a < N//2] = -1
    # create ak version
    b = ak.arange(N)
    b[b < N//2] = -1
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

print("check (expanding) set bool iv = value:", check_set_bool_iv_value(N))

def check_set_bool_iv(N):
    # create np version
    a = np.arange(N)
    a[a < N//2] = a[:N//2] * -1
    # create ak version
    b = ak.arange(N)
    b[b < N//2] = b[:N//2] * -1
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

print("check (expanding) set bool iv = pda:", check_set_bool_iv(N))

def check_get_integer_iv(N):
    # create np version
    a = np.arange(N)
    iv = np.arange(N//2)
    a = a[iv]
    # create ak version
    b = ak.arange(N)
    iv = ak.arange(N//2)
    b = b[iv]
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

print("check (gather) get integer iv:", check_get_integer_iv(N))

def check_set_integer_iv_value(N):
    # create np version
    a = np.arange(N)
    iv = np.arange(N//2)
    a[iv] = -1
    # create ak version
    b = ak.arange(N)
    iv = ak.arange(N//2)
    b[iv] = -1
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

print("check (scatter) set integer iv = value:", check_set_integer_iv_value(N))

def check_set_integer_iv(N):
    # create np version
    a = np.arange(N)
    iv = np.arange(N//2)
    a[iv] = iv*10
    # create ak version
    b = ak.arange(N)
    iv = ak.arange(N//2)
    b[iv] = iv*10
    # print(a,b)
    c = a == b.to_ndarray()
    # print(type(c),c)
    return pass_fail(c.all())

print("check (scatter) set integer iv = pda:", check_set_integer_iv(N))

def check_get_integer_idx(N):
    # create np version
    a = np.arange(N)
    v1 = a[N//2]
    # create ak version
    b = ak.arange(N)
    v2 = b[N//2]
    return pass_fail(v1 == v2)

print("check get integer idx :", check_get_integer_idx(N))

def check_set_integer_idx(N):
    # create np version
    a = np.arange(N)
    a[N//2] = -1
    v1 = a[N//2]
    # create ak version
    b = ak.arange(N)
    b[N//2] = -1
    v2 = b[N//2]
    return pass_fail(v1 == v2)

print("check set integer idx = value:", check_set_integer_idx(N))

if args.shutdown_server:
    ak.shutdown()
else:
    ak.disconnect()
sys.exit(errors)
