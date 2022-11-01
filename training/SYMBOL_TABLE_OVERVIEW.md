# Overview

The purpose of this document is to serve as a high-level guide to the structure and elements within Arkouda's Symbol Table. This guide does not cover creating or refactoring new Symbol Table Entry types. 

<a id="toc"></a>
## Table of Contents
1. [Symbol Table Overview](#overview)
2. [Symbol Table Structure](#symStruct)
3. [Creating a New Entry From an Array](#addSym)
4. [Finding an Entry in the Symbol Table](#findSym)
5. [Accessing Symbol Table Entry Data](#accessSym)

<a id="overview"></a>
## Symbol Table Overview

The Symbol Table is Arkouda's "database" that tracks the components every Arkouda object is composed of. At a fundamental level it functions similar to a relational database, however the design and access methods are quite different.

<a id="symStruct"></a>
## Symbol Table Structure

The Symbol Table stores `Entries`. The Entry types are designed in a simple hierarchical structure as follows:

- `AbstractSymEntry`
  - `GenSymEntry`
    - `SymEntry`
    - `SegArraySymEntry`
    - `SegStringSymEntry`
  - `CompositeSymEntry`
    - `GroupBySymEntry`
  
The root-level entry type is `AbstractSymEntry`. All other entry types inherit from `AbstractSymEntry` or one of its ancestors. `AbstractSymEntry` does not store any data, only metadata information, such as name and entry type.

`GenSymEntry` is the first type directly inheriting from `AbstractSymEntry`. `GenSymEntry` is used to define the main metadata necessary for entry types that contain a single set of values of a single data type.

`SymEntry` is the entry type used for basic one-dimensional arrays of numeric values and their metadata. This is the most commonly used entry type.

`SegArraySymEntry` tracks the component pieces of a `SegmentedArray` object. These components are all instances of `SymEntry` types.

`SegStringSymEntry`tracks the component pieces of a `SegmentedStrings` object. The values of a `SegStringSymEntry` are stored in a `uint8` `SymEntry`. These components are all instances of `SymEntry` types.

`CompositeSymEntry` is used to indicate that child `Entries` inheriting from this entry type contain multiple `data-adding` objects. This means that types under `CompositeSymEntry` contain multiple objects of varying types. For example, `SegArraySymEntry` contains three `SymEntry` entries, but all three are of a single type and relate to a single SegArray Object. `GroupBySymEntry` contains five `SymEntry` entries, but each `SymEntry` could be of a different type from the others, i.e. `keyNamesEntry` would be of type `str` while `permEntry` would be of type `int64`.

`GroupBySymEntry` is used to track component pieces of a `GroupBy` object. It accomplishes this task by tracking five separate `SymEntry` instances that each relate to one of the components.

<a id="addSym"></a>
## Creating a New Entry From an Array

If your Chapel module creates an array of data that you then want to either persist or return a reference to back to the Client, then you would need to add that array to the symbol table. This is pretty simple and only requires a single line of code to accomplish.

```chapel
  // Assume we have a borrowed Symbol Table (SymTab) called 'st'
  
  // Array Creation
  var myArray: [0..#4] int = [1, 2, 3, 4, 5];
  
  // Get the next available name in the symbol table and the add the entry using that name
  var newName = st.nextName();
  st.addEntry(newName, new shared SymEntry(myArray));
```

Once added, that array can then be accessed from the symbol table using the value of `newName`.

<a id="findSym"></a>
## Finding an Entry in the Symbol Table

There are two ways of finding if a given string matches the name of a Symbol Table entry:

```chapel
  st.contains(string); // Returns bool
  st.findAll(regex string); // Returns [] string
```

`st.contains(string)` is used to verify the existence of an entry that matches the provided string. This function will return `True` if it exists within the Symbol Table and `False` if it does not.

`st.findAll(regex string)` is used when you have a partial name that can map to multiple entries in a set. An example of when this is used is when `Categorical` objects are registered to the symbol table. These objects register 4 separate entries, one for each piece of the `Categorical` needed to rebuild it on the client. 

If you have a set of data with entries `dataset_entry_0`, `dataset_entry_1`, and `dataset_entry_2`, you could manually find each entry if you know how many entries are in your dataset. However, this function is most useful for when you may not know exactly how many entries you have. By passing the portion of the name these three entries have in common, `dataset_entry_`, followed by regex notation for "Any number, 1 or more times", `/d+`. We can then get a list of all the matching names in the symbol table and operate on all the dataset pieces.

```chapel
  var entryList = st.findAll("dataset_entry_/d+");
  // entryList: [`dataset_entry_0`, `dataset_entry_1`, 'dataset_entry_2']
```

You can then loop through the returned array of names to access the data and perform operations on each entry in the set.

<a id="accessSym"></a>
## Accessing Symbol Table Entry Data

Depending on your functionality requirements, there are a couple different ways to access a symbol table entry.

### Passing the Entry to the Client

If you need to return an entry to the python for computations, the `SymTab.attrib(name)` method will return the entry's attributes, which are required by the arkouda client to reference the entry and use the underlying data. You can then use the return message to `attach` to the entry and access the data as you would with any other Python object.

More information on passing data between the client and server can be found [here](https://github.com/Bears-R-Us/arkouda/blob/master/training/MESSAGING_OVERVIEW.md).

### Working with Entries in Chapel

When working with a basic `SymEntry` entry type, you can easily access the data values within the entry by accessing the property `a`. To access the domain information, access the property `aD`. For example, assume we have a SymTab entry variable called `entry`.

```chapel
  var values = entry.a;
  var domain = entry.aD;
```

Each entry type has its own set of properties that can be accessed using the `.` notation, similar to accessing class properties in Python objects. Below is a list of all accessible properties for each entry type within the hierarchy:

- `AbstractSymEntry`
  - `entryType`
  - `assignableTypes`
  - `name`
  - `GenSymEntry`
    - All properties of `AbstractSymEntry`
    - `dtype`
    - `itemsize`
    - `size`
    - `ndim`
    - `shape`
    - `SymEntry`
      - All properties of `GenSymEntry`
      - `etype` - Chapel's version of `dtype`
      - `a` - Stores the values of the array
      - `aD` - Stores the domain information
    - `SegArraySymEntry`
      - All properties of `GenSymEntry`
      - `segmentsEntry`
      - `valuesEntry`
      - `lengthsEntry`
    - `SegStringSymEntry`
      - All properties of `GenSymEntry`
      - `offsetsEntry`
      - `bytesEntry`
  - `CompositeSymEntry`
    - All properties of `AbstractSymEntry`
    - `ndim`
    - `size`
    - `GroupBySymEntry`
      - All properties of `CompositeSymEntry`
      - `keyNamesEntry`
      - `keyTypesEntry`
      - `segmentsEntry`
      - `permEntry`
      - `ukIndEntry`

An example of accessing these properties is as follows. Assume we have a `SegArraySymEntry` called segArray:

```chapel
  // `valuesEntry` stores a `SymEntry`, so we use `.a` to get the values of `valuesEntry`  
  var values = segArray.valuesEntry.a
```
