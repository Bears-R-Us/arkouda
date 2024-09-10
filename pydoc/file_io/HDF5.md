# HDF5

HDF5 is an extremely flexible format. Because of this, it is important to adhere to these specifications in order for a file to be readable by Arkouda.

More information on HDF5 is available [here](https://www.hdfgroup.org/solutions/hdf5/).

## File Configuration

Arkouda supports saving HDF5 files in 2 ways:

- Single File
  - All data is pulled local to the processing root node and saved into one file
- Distributed Files (Default)
  - Each file contains the portion of the data local to the locale where the file is being written. This results in one file per locale.

*It is important to note that the file schemas are the same in both cases.*

## Supported Arkouda Data Types

While most objects in Arkouda can be saved, there are 3 main datatypes currently supported within HDF5.

- pdarray
- Strings
- DataFrame
- Index
- Categorical
- SegArray
- GroupBy

HDF5 is able to contain any number of objects within the same file.

## MetaData Attributes

All data within the HDF5 file is expected to contain several attributes that aide in determining the data within the object. These attributes are assigned at the `Group` and `Dataset` levels.

`ObjType`: `int`
> Integer representing the type of object stored in the group/dataset. This corresponds to the Chapel `enum ObjType`. Required to properly read each object.
>
- 0 = `ArrayView` (Deprecated)
- 1 = `pdarray`
- 2 = `Strings`
- 3 = `SegArray`
- 4 = `Categorical`
- 5 = `GroupBy`

`isBool`: `int`
> Integer value (0 or 1) representing a boolean value that indicates if the data stored contains boolean values. This is only required to be set when the dataset contains boolean values.

`file_version`: `real(32)` (Optional)
> Real value indicating the formatting version. `0.0` and `1.0` are no longer in use. Should be `2.0`.

`arkouda_version`: `c_string` (Optional)
> String value of the Arkouda version at the time the object was written.

The 2 attributes marked `Optional` are not required for data to be read. Thus, if you are reading data into Arkouda from another source, these can be omitted. However, any dataset written out by Arkodua will include this information.

*Additional object types are being worked for direct support.*

## Data Schema

This section provides an outline of the expected data schema for each object type. Each example assumes the top level group/dataset is not nested.

When reading array values, the data type of the values is automatically detected and is therefore not required to be included in the metadata.


### pdarray

> 1. Dataset (will have a user provided name. Defaults to 'array')
>       1. Attributes
>           1. ObjType: 1
>           2. isBool: 0 or 1
>           3. file_version: 2.0 (Optional)
>           4. arkouda_version: 'current_arkouda_version' (Optional)
>       2. Data - values of the pdarray.

### Strings

`Strings` objects are stored within an HDF5 group. This group contains datasets storing the values and segments separately.

>1. Group (user provided dataset name. Defaults to 'strings_array')
>       1. Attributes
>           1. ObjType: 2
>           2. file_version: 2.0 (Optional)
>           3. arkouda_version: 'current_arkouda_version' (Optional)
>       2. Dataset - Values (user provided dataset name with `_values` appended)
>           1. Attributes
>               1. ObjType: 1
>               2. isBool: 0 or 1
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - uint8 values representing our string values. Includes null byte termination.
>       3. Dataset - Offsets (user provided dataset name with `_segments` appended) (Optional)
>           1. Attributes
>               1. ObjType: 1
>               2. isBool: 0
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - int64 values representing in start index of each string value.

*Please Note - The offsets dataset is not required but can be provided. Strings uses null byte termination and is able to calculate the offsets of its components during reads.*

### SegArray

`SegArray` objects are stored within an HDF5 group. This group contains datasets storing the values and segments separately.

>1. Group (user provided dataset name. Defaults to 'segarray')
>       1. Attributes
>           1. ObjType: 3
>           2. file_version: 2.0 (Optional)
>           3. arkouda_version: 'current_arkouda_version' (Optional)
>       2. Dataset - Values
>           1. Attributes
>               1. ObjType: 1 or 2
>               2. isBool: 0 or 1
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - numeric values representing our string values. int64, uint64, float64, or bool.
>       3. Dataset - Offsets
>           1. Attributes
>               1. ObjType: 1
>               2. isBool: 0
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - int64 values representing the start index of each segmented value.

### Categorical

`Categorical` objects are stored within an HDF5 group. This group contains datasets storing the components of the Categorical.

>1. Group (user provided dataset name. Defaults to 'categorical')
>       1. Attributes
>           1. ObjType: 4
>           2. file_version: 2.0 (Optional)
>           3. arkouda_version: 'current_arkouda_version' (Optional)
>       2. Dataset - Codes
>           1. Attributes
>               1. ObjType: 1
>               2. isBool: 0
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - int64 values representing our codes of the Categorical.
>       3. Dataset - Categories
>           1. Attributes
>               1. ObjType: 2
>               2. isBool: 0
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - Strings group representing the categories of the Categorical.
>       4. Dataset - NA_Codes
>           1. Attributes
>               1. ObjType: 1
>               2. isBool: 0
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - int64 values representing the index in of categories with NA value.
>       5. Dataset - Permutation (Optional. Only include if Categorical object has permutation property)
>           1. Attributes
>               1. ObjType: 1
>               2. isBool: 0
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - int64 values representing the permutation of the categories.
>       6. Dataset - Segments (Optional. Only include if Categorical object has segments property)
>           1. Attributes
>               1. ObjType: 1
>               2. isBool: 0
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - int64 values representing the start index of category segments.

### GroupBy

`GroupBy` objects are stored within an HDF5 group. This group contains datasets storing the components of the GroupBy.

>1. Group (user provided dataset name. Defaults to 'groupby')
>       1. Attributes
>           1. ObjType: 5
>           2. file_version: 2.0 (Optional)
>           3. arkouda_version: 'current_arkouda_version' (Optional)
>       2. Dataset - Permutation
>           1. Attributes
>               1. ObjType: 1
>               2. isBool: 0
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - int64 values representing the permutation of the GroupBy.
>       3. Dataset - Segments
>           1. Attributes
>               1. ObjType: 1
>               2. isBool: 0
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - int64 values representing the start index of GroupBy segments.
>       4. Dataset - unique_key_idx
>           1. Attributes
>               1. ObjType: 1
>               2. isBool: 0
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - int64 values representing the index of Unique keys in the GroupBy.
>       5. Dataset - KEY_# (multiple keys may be present. They will be numbered accordingly)
>           1. Attributes
>               1. ObjType: 1, 2, or 4 (pdarray, Strings, or Categorical)
>               2. isBool: 0 or 1
>               3. file_version: 2.0 (Optional)
>               4. arkouda_version: 'current_arkouda_version' (Optional)
>           2. Data - Key object used to generate the GroupBy. This will be a dataset or group depending on the object type.

## Supported Write Modes

**Truncate**
> When writing to HDF5 in `truncate` mode, any existing HDF5 file with the same name will be overwritten. If no file exists, one will be created. If writing multiple objects, the first is written in `truncate` mode. All subsequent objects will then be appended to the file. The user will be notified of any overwritten files.

**Append**
> When writing to HDF5 in `append` mode, all datasets will be appended to the file. If no file with the supplied name exists, one will be created. If any datasets being written have a name that is already the name of a dataset within the file, an error will be generated.

## Data Distribution

**Single File**
> If the user elects to write to a single HDF5 file, all data will be pulled to the processing node and saved to ONE file with the supplied file name. It is important to ensure that the object is small enough to prevent memory exhaustion on the node.

**Distributed Files**
> If the user elects to write data to distributed files, data will be written to one file per locale. Each file will contain the data from the object local to the locale of that file. File names will be the name provided by the user with the suffix `_LOCALE####` where `####` will be replaced with the locale number. Because the data is distributed across multiple nodes, there is a much lower risk of memory exhaustion.

## Legacy File Support

Older version of Arkouda used different schemas for `pdarray` and `Strings` objects (`ArrayView` was not supported). This format does not include the explicit `ObjType` attribute and requires the type to be inferred during processing. Reading these files is still supported by Arkouda. When the data type is `uint8` and the object with the name `dataset` (user supplied dataset name) is a group containing a dataset name `values` the object is assumed to be of object type Strings.

## API Reference

### pdarray

```{eval-rst}  
- :py:meth:`arkouda.pdarray.to_hdf`
- :py:meth:`arkouda.pdarray.save`
```

### Index

```{eval-rst}  
- :py:meth:`arkouda.Index.to_hdf`
- :py:meth:`arkouda.Index.save`
```

### DataFrame

```{eval-rst}  
- :py:meth:`arkouda.DataFrame.to_hdf`
- :py:meth:`arkouda.DataFrame.save`
- :py:meth:`arkouda.DataFrame.load`
```

### Strings

```{eval-rst}  
- :py:meth:`arkouda.Strings.to_hdf`
- :py:meth:`arkouda.Strings.save`
```

### Categorical

```{eval-rst}  
- :py:meth:`arkouda.Categorical.to_hdf`
- :py:meth:`arkouda.Categorical.save`
```

### SegArray

```{eval-rst}  
- :py:meth:`arkouda.SegArray.to_hdf`
- :py:meth:`arkouda.SegArray.load`
```

### GroupBy

```{eval-rst}  
- :py:meth:`arkouda.GroupBy.to_hdf`
```
