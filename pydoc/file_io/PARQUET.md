# Parquet

Parquet is a column-oriented file format that provides more structure than HDF5. While this is extremely beneficial, it does have some limitations within Arkouda at this time due to the requirement that columns have equal sizes.

*We are currently working on providing functionality that eliminates these limitations in order to support more data types being saved to Parquet.*

More information on Parquet can be found [here](https://parquet.apache.org/).

## Supported Arkouda Data Types

- pdarray
- Index
- DataFrame
- Strings
- SegArray
  - Strings values are only supported in the single instance case. Track updates on support when writing multiple columns [here](https://github.com/Bears-R-Us/arkouda/issues/2493)

## Compression

Parquet supports 5 compression types:

- Snappy
- GZip
- Brotli
- ZSTD
- LZ4

Data can also be saved using no compression. Arkouda now supports writting Parquet files with all compression types supported by Parquet.

## Supported Write Modes

**Truncate**
> When writing to Parquet in `truncate` mode, any existing Parquet file with the same name will be overwritten. If no file exists, one will be created. If writing multiple objects, all corresponding columns will be written to the Paruqet file at once.

**Append**
> When writting to Parquet in `append` mode, all datasets will be appended to the file. If no file with the supplied name exists, one will be created. If any datasets being written have a name that is already the name of a dataset within the file, an error will be generated. Append is not supported for SegArray objects.
>
>*Please Note: appending to a Parquet file is not natively support and is extremely ineffiecent. It is recommended to read the file out and call `arkouda.pandas.io.to_parquet` on the output with the additional columns added and then writting in `truncate` mode.*

## API Reference

### pdarray

```{eval-rst}  
- :py:meth:`arkouda.pdarray.to_parquet`
- :py:meth:`arkouda.pdarray.save`
```

### Index

```{eval-rst}  
- :py:meth:`arkouda.Index.to_parquet`
- :py:meth:`arkouda.Index.save`
```

### DataFrame

```{eval-rst}  
- :py:meth:`arkouda.pandas.DataFrame.to_parquet`
- :py:meth:`arkouda.pandas.DataFrame.save`
- :py:meth:`arkouda.pandas.DataFrame.load`
```

### Strings

```{eval-rst}  
- :py:meth:`arkouda.numpy.Strings.to_parquet`
- :py:meth:`arkouda.numpy.Strings.save`
```

### SegArray

```{eval-rst}  
- :py:meth:`arkouda.numpy.SegArray.to_parquet`
```

### Categorical

Categorical objects cannot currently be written to Parquet Files. This is due to the fact that the components of Categoricals can have different sizes.
