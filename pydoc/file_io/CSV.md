# CSV

Arkouda now has support for reading and writing CSV file formats. CSV will not perform at the same level as HDF5 and Parquet. It is intended for interacting with smaller datasets and will prevent the need to convert files already in CSV format to HDF5 or Parquet.

## Support Arkouda Data Types

- pdarray
- Strings
- Index
- DataFrame

## File Formatting

Arkouda supports reading CSV files of various formats. Current limitations include:

- All lines/rows of the CSV file must be newline (`\n`) delimited
- It is assumed that all files contain column names. The column names should be the first line of data in the file.
- Files written by Arkouda will contain a "header" with typing information for the columns. Files without this header will return all read data as Strings objects.
- Custom column delimiters can be used. The default column delimiter is ",". The column delimiter set for the file will also be used to delimit the column names.
- Header contents are always comma (`,`) delimited.

### Example Files

To give an idea of arkouda supported formats for CSV files, we have provided two example files that Arkouda can read: one written by arkouda and one written outside of Arkouda.

#### Arkouda Formatted File

```text
**HEADER**
int64,str,float64
*/HEADER/*
ColA,ColB,ColC
0,ABC,3.14
1,DEF,5.56
2,GHI,2.11
```

#### File Without Header

Arkouda can read files without the header. All data will be read out to Strings objects in this case.

```text
ColA,ColB,ColC
0,ABC,3.14
1,DEF,5.56
2,GHI,2.11
```

## Data Formatting

Because CSV is a text format, all data is stored as a string. If the header is provided, the pdarray resulting from a read will be the assigned type.

CSV files have one major difference in how they store data in comparison to HDF5 and Parquet, specifically for Strings objects. CSV stores Strings objects as the actual string, not as a `uint(8)` array as in HDF5 and Parquet.

## API Reference

Due to differences in execution of the CSV format, generic load/read functionality is not currently supported. As a result, the provided `read_csv` methods must be used at this time.

`ls`/`get_dataset` functionality is supported, but is once again separate from the generic implementation. To return a list of columns in the CSV use the `ls_csv()`/`ak.get_columns()` function.

### pdarray

```{eval-rst}  
- :py:meth:`arkouda.pdarray.to_csv`
```

### Strings

```{eval-rst}  
- :py:meth:`arkouda.Strings.to_csv`
```

### Index

```{eval-rst}  
- :py:meth:`arkouda.Index.to_csv`
```

### DataFrame

```{eval-rst}  
- :py:meth:`arkouda.DataFrame.to_csv`
- :py:meth:`arkodua.DataFrame.read_csv`
```
