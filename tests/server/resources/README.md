# Unit Test Resources
This directory contains sample files used for Chapel-based unit tests.

### File: sample.ascii.txt
A basic ascii text file containing the numbers `123456789`

### File: sample.hdf5
A sample HDF5 file of 100 zeros generated via
```python
import h5py
f = h5py.File("sample.hdf5", "w")
data = f.create_dataset("sample", (100,), dtype="i")
f.close()
```

### File: sample.parquet
A sample Apache parquet file consisting of the numbers 1-5 in a column named "numbers" generated via
```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

s = pd.Series([1, 2, 3, 4, 5])
df = pd.DataFrame({"numbers":s})
t = pa.Table.from_pandas(df)
pq.write_table(t, "sample.parquet")
```

### File: sample.arrow
A sample Apache-Arrow IPC / File object generated via (adapted from https://arrow.apache.org/docs/python/ipc.html)
```python
import pyarrow as pa
data = [ pa.array([1, 2, 3, 4, 5]) ]
batch = pa.record_batch(data, names=['f0'])
writer = pa.ipc.new_file("sample.arrow", batch.schema)
writer.write_batch(batch)
writer.close()
```