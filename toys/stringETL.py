import numpy as np
import h5py
import pandas as pd

# df = pd.read_csv('/mnt/data/lanl_netflow/netflow_day-02', nrows=10000, names=['start', 'duration', 'srcIP', 'dstIP', 'proto', 'srcPort', 'dstPort', 'srcPkts', 'dstPkts', 'srcBytes', 'dstBytes'])

def pack_strings(column):
    lengths = column.apply(len).values
    offsets = lengths.cumsum() + np.arange(lengths.shape[0]) - lengths
    totalbytes = lengths.sum() + lengths.shape[0]
    packed = np.zeros(shape=(totalbytes,), dtype=np.uint8)
    for (o, s) in zip(offsets, column.values):
        for i, b in enumerate(s.encode()):
            packed[o+i] = b
    return packed, offsets
            
def write_strings(filename, group, packed, offsets, mode='w'):
    with h5py.File(filename, mode) as f:
        g = f.create_group(group)
        g.attrs["segmented_string"] = np.bool(True)
        g.create_dataset('values', data=packed)
        g.create_dataset('segments', data=offsets)

