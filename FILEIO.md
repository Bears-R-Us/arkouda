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

### File Formats

HDF5 now supports saving datasets in 2 different file configurations. 

- Single File
- Distributed Files (Default)

When saving to a single file, all the data from an Arkouda object is stored to one file. This file is stored on `LOCALE0`. When saving data to a distributed file system, data is stored in one file per `LOCALE`. Each file contains the portion of data from the object local to the `LOCALE` the file is being written to. Each file can contain multiple datasets/groups and thus can store multiple objects.

### MetaData Attributes

These attributes are required to be set for each group and dataset.

`ObjType`: `int`
   Integer representing the type of object stored in the group/dataset. This corresponds to the Chapel `enum ObjType`. Required to properly read each object. 
   - 0 = `ArrayView`
   - 1 = `pdarray`
   - 2 = `Strings`

`isBool`: `int`
   Integer value (0 or 1) representing a boolean value that indicates if the data stored contains boolean values. This is only required to be set when the dataset contains boolean values.

`file_version`: `real(32)`
   Real value indicating the formatting version. `0.0` and `1.0` are no longer in use. Should be `2.0`.

`arkouda_version`: `c_string`
   String value of the Arkouda version at the time the object was written.

### Supported Arkouda Data Types

While most objects in Arkouda can be saved, there are 3 main datatypes currently supported within HDF5.

- pdarray
- Strings
- ArrayView (Import/Export not Supported) 

### PDArray/ArrayView Dataset Format
`ArrayView` and `pdarray` objects' storage format is identical. The only difference is that `ArrayView` objects require additional attributes to ensure that they can be read properly. These objects are stored in an HDF5 dataset. 

**Structure**
1) Dataset
   1) Data - ArrayView/pdarray values
   2) Attributes
      1) MetaData Attributes
      2) ArrayView Attributes (If the `ObjType` is equivalent to `ArrayView`)

**ArrayView Attributes**

`Rank`: `int` 
   Integer representing the number of dimensions in the dataset. This should be stored as the rank of the *unflattened* data, even when storing as a flattened array.

`Shape`: `int array` Integer array storing the size of each dimension. The array should be of length equal to the `Rank`.

### Strings DataSet Format
`Strings` objects are stored within an HDF5 group. This group contains datasets storing the values and segments separately. 

**Structure**
1) Group
   1) Dataset - `values`
      1) `ObjType` Attribute
      2) Data - String object's values pdarray data
   2) DataSet - `segments`
      1) `ObjType` Attribute
      2) Data - String object's segments pdarray data
   3) MetaData Attributes

Each dataset within the group contains the `ObjType` attribute so that they can be read individually as a dataset. The `isBool` attribute is not needed because these objects will never store boolean values.

## Parquet
COMING SOON

## Reading Objects
Arkouda objects can be read from files using the `ak.read()` or `ak.load()` functions. More information on these functions are linked below.

- [ak.read](https://bears-r-us.github.io/arkouda/autoapi/arkouda/pdarrayIO/index.html#arkouda.pdarrayIO.read)
- [ak.load](https://bears-r-us.github.io/arkouda/usage/IO.html#arkouda.load)
- [ak.load_all](https://bears-r-us.github.io/arkouda/usage/IO.html#arkouda.load_all)

## Writing Objects
*Objects currently being written with file version `v2.0`.*

Arkouda objects can be written to files using the `ak.obj.save()` or `ak.save_all()` functions.

- [ak.save_all](https://bears-r-us.github.io/arkouda/autoapi/arkouda/pdarrayIO/index.html#arkouda.pdarrayIO.save_all)

Additionally, there are `save` functions for individual Arkouda objects. The function definition is detailed below as it is the same for each object type.

```python
def save(self, filepath: str, dset: str, mode: str = "truncate", file_type: str = "distribute")
    """
        Save the current object to hdf5 file
        Parameters
        ----------
        filepath: str
            Path to the file to write the dataset to
        dset: str
            Name of the dataset to write
        mode: str (truncate | append)
            Default: truncate
            Mode to write the dataset in. Truncate will overwrite any existing files.
            Append will add the dataset to an existing file.
        file_type: str (single|distribute)
            efault: distribute
            Indicates the format to save the file. Single will store in a single file.
            Distribute will store the date in a file per locale.
```