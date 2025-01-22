import arkouda as ak

ak.connect()

size = 10**8

a = ak.randint(0, 2**32, size)
b = ak.randint(0, 2**32, size)

df = ak.DataFrame({"col1": a, "col2": b})

df.to_parquet("test-file")
