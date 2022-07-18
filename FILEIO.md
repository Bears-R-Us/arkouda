# Arkouda File Support

*Please Note: This file is being developed in conjunction with updates to our file I/O system. Information is being omitted until updates on each section are completed to avoid confusion.*

## Table of Contents
1. [File Types](#filetypes)
2. [HDF5](#hdf)
   1. [Multidimensional Objects](#multidim)

## Supported File Types
Arkouda currently supports the file types listed below. The way the data is stored may vary. This file will detail the "schema" each file type is expected to follow. If your file does not follow the detailed "schema", please try using our `import`/`export` tools. *Please Note: The functionality of the `import`/`export` tools is dependent on the size of the data because they only run on the client.*

- HDF5
- Parquet

## HDF5

**Simplified Structure**
1) File
   1) Dataset
      1) Attributes
      2) Data

Each file can contain multiple datasets. Groups are not currently used.

### Multidimensional Objects

*Please Note: `import`/`export` functionality is not currently supported for multidimensional objects.*

1) File
   1) Dataset 
      1) Attribute: `Rank`
      2) Attribute: `Shape`
      3) Attribute: `ObjType`
      4) Attribute: `Format`
      5) Data

`Rank`: `int` 
   Integer representing the number of dimensions in the dataset. This should be stored as the rank of the *unflattened* data, even when storing as a flattened array.

`Shape`: `int array` Integer array storing the size of each dimension. The array should be of length equal to the `Rank`.

`ObjType`: `b'ArrayView'` Byte string indicating the object type. This is used when loading data to return the correct object type.

`Format`: `int` Integer that is mapped to `flat` and `multi`. `flat==0` and `multi==1`. This indicates the formatting of the data within the HDF5 dataset.
