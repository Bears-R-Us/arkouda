# Overview

The purpose of this document is to serve as a high-level guide to the structure and elements within Arkouda's Symbol Table. This guide does not cover creating or refactoring new Symbol Table Entry types. 

For more information regarding the creation of new Symbol Table Entry types, view the [Symbol Table Entry Creation](https://github.com/Bears-R-Us/arkouda/blob/master/training/SYMBOL_TABLE_ENTRY_CREATION.md) document.

<a id="toc"></a>
## Table of Contents
1. [Symbol Table Overview](#overview)
2. [Symbol Table Structure](#symStruct)
3. [Creating a New Entry From an Array](#addSym)
4. [Finding an Entry in the Symbol Table](#findSym)
5. [Accessing Symbol Table Entry Data](#accessSym)

<a id="overview"></a>
## Symbol Table Overview

The Symbol Table is implemented using a Chapel Map and serves as a `lookup table` which allows Arkouda to persist objects until the server is shutdown or the object is deleted.


<a id="symStruct"></a>
## Symbol Table Structure

Records stored in the Symbol Table are called `Entries`. The Entry types are designed in a simple hierarchical structure as follows:

- `AbstractSymEntry`
  - `GenSymEntry`
    - `SymEntry`
    - `SegArraySymEntry`
    - `SegStringSymEntry`
  - `CompositeSymEntry`
    - `GroupBySymEntry`
  
The root-level entry type is `AbstractSymEntry`. All other entry types inherit from `AbstractSymEntry` directly or from one of its children. `AbstractSymEntry` does not store any data, only metadata information, such as name and entry type.

`GenSymEntry` is the first type directly inheriting from `AbstractSymEntry`. `GenSymEntry` is used to define the main metadata necessary for entry types that contain a single set of values and its necessary components. For example, a `SegArraySymEntry` stores three `SymEntries`, `values`, `segments`, and `lengths`, however these three pieces are all necessary to build a single instance of a `SegArray` object and reference a single set of data values.

`SymEntry` is the entry type used for basic one-dimensional arrays of numeric values and their metadata. This is the most commonly used entry type.

`SegArraySymEntry` tracks the component pieces of a `SegmentedArray` object. These components are all instances of `SymEntry` types.

`SegStringSymEntry`tracks the component pieces of a `SegmentedStrings` object. The values of a `SegStringSymEntry` are stored in a `uint8` `SymEntry`. These components are all instances of `SymEntry` types.

`CompositeSymEntry` is used to indicate that child `Entries` inheriting from this entry type contain multiple `data-adding` objects. This means that entry types under CompositeSymEntry contain multiple objects of varying types, which may not all be needed to conceptualize the object. For example, `GroupBySymEntry` contains five `SymEntry` entries. One of these, `keyNamesEntry` could contain `n` number of references to additional data values. These could be of multiple types, and it is possible to build a GroupBy object with all or only part of the stored values.

`GroupBySymEntry` is used to track component pieces of a `GroupBy` object. It accomplishes this task by tracking five separate `SymEntry` instances that each correspond to one of the components.

<a id="addSym"></a>
## Creating a New Entry From an Array

If your Chapel module creates an array of data that you then want to either persist or return a reference to back to the Client, then you would need to add that array to the symbol table. This is pretty simple to accomplish.

```chapel
  // Assume we have a borrowed Symbol Table (SymTab) called 'st'
  
  // Array Creation
  var myArray: [0..#4] int = [1, 2, 3, 4, 5];
  
  // Get the next available name in the symbol table and the add the entry using that name
  var newName = st.nextName();
  st.addEntry(newName, new shared SymEntry(myArray));
```

This process is assigning a code-generated name to the array and creating a link using that name to the data passed in `myArray`. This means that you can then use the name generated and stored in `newName` to access the array data anywhere.

<a id="findSym"></a>
## Finding an Entry in the Symbol Table

There are three ways of finding if a given string matches the name of a Symbol Table entry:

```chapel
  st.checkTable(string); // Throws an exeption if not found
  st.contains(string); // Returns bool
  st.findAll(regex string); // Returns [] string
```

`st.checkTable(string)` is the most commonly used method of ensuring an entry exits for the given name. In most cases this is used to prevent attempting to perform operations on a non-existent Symbol Table Entry and throw an error because the entry should exist.

`st.contains(string)` is used to verify the existence of an entry that matches the provided string. This function will return `True` if it exists within the Symbol Table and `False` if it does not. This is useful if your function does not necessarily expect an entry to exist, but should use the entry if it does.

`st.findAll(regex string)` is currently only used in `register/attach` functionality as it was specifically designed to handle cases of `attaching` to multiple-component data objects that don't have a Chapel equivalent, such as `Series` or `DataFrame`, easily. It is used when you have a partial name that can map to multiple entries in a set. An example of when this is used is when `Categorical` objects are registered to the symbol table. These objects register 4 separate entries, one for each piece of the `Categorical` needed to rebuild it on the client. 

If you have a set of data with entries `dataset_entry_0`, `dataset_entry_1`, ... `dataset_entry_n` and you don't know how many entries are in this set, this function makes it easy to find all of the entries. By passing the portion of the name these entries have in common, `dataset_entry_`, followed by regex notation for "Any number, 1 or more times", `/d+`. We can then get a list of all the matching names in the symbol table and operate on all the dataset pieces.

```chapel
  var entryList = st.findAll("dataset_entry_/d+");
  // entryList: [`dataset_entry_0`, `dataset_entry_1`, 'dataset_entry_2']
```

You can then loop through the returned array of names to access the data and perform operations on each entry in the set.

<a id="accessSym"></a>
## Accessing Symbol Table Entry Data

Depending on your functionality requirements, there are a couple different ways to access a symbol table entry.

### Passing the Entry to the Client

If you need to return an entry to python, the `SymTab.attrib(name)` method will return the entry's metadata, which is required by the arkouda client to reference the entry when operations are sent to the server to be performed on the object. You can then use the return message to `attach` to the entry and view the data as you would with any other Python object.

It is important to note that on the client after `attaching` to a Symbol Table Entry, the client does not store the data or perform computations on the data. The client is only storing the metadata and any computations on the object are performed on the server.

More information on passing data between the client and server can be found [here](https://github.com/Bears-R-Us/arkouda/blob/master/training/MESSAGING_OVERVIEW.md).

### Working with Entries in Chapel

When working with a basic `SymEntry` entry type, you can easily access the data values within the entry by accessing the property `a`. To access the domain, access the property `aD`. For example, assume we have a SymTab entry variable called `entry`.

```chapel
  var values = entry.a;
  var entryDomain = entry.aD;
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
