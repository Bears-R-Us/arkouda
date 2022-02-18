import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import pyarrow as pa

df = pd.DataFrame({'one': ['asd','fgh','jkl'], 'two': ['thi','sco','lum']});
table = pa.Table.from_pandas(df)

pq.write_table(table, 'str-file.parquet')
