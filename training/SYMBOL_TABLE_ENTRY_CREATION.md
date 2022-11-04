# Creating a New Symbol Table Entry

This guide explains how to add a new Entry type to the Symbol Table. There are two main components:

- The configuration of the Symbol Table entry.
  - This class is what is stored in the Symbol Table and is used to persist the object on the server.
- The object used during processing.
  - This object is generated from the Symbol Table entry.

## Table of Contents
1. [Determining Parent Class](#parent)
   1. [When to use `GenSymEntry`](#parent-gen)
   2. [When to use `CompositeSymEntry`](#parent-comp)
   3. [Still Unsure of which Parent Class to Use?](#unsure)
2. [Building the New Symbol Table Entry](#build)
   1. [Add to SymEntryType Enum](#enum)
   2. [Adding Properties](#props)
   3. [Initializing the Object](#init)
   4. [Providing the Size Estimation](#sizeEst)
   5. [Handle Dynamic Types](#dynamic)
3. [Creating the Object](#create)

<a id="parent"></a>
## Determining Parent Class

Arkouda has two main parent classes for Symbol Table Entries, `GenSymEntry` and `CompositeSymEntry`. All Symbol Table entries inherit from `AbstractSymEntry`. This is extremely important for allowing generic handling during processing. We will not be documenting the process of adding a Symbol Table Entry that directly inherits from `AbstractSymEntry` because in all anticipated cases, leveraging `GenSymEntry` or `CompositeSymEntry` should suffice. 

<a id="parent-gen"></a>
### When to use `GenSymEntry`
`GenSymEntry` objects all contain the properties listed below:

- `dtype`: The data type the object contains.
- `itemsize`: The size of an individual data point.
- `size`: The number of data points in the object.
- `ndim`: The number of dimensions in the object.
- `shape`: Integer array of size `ndim` detailing the number of elements in each dimension.

`GenSymEntry` is the parent class of all server objects in Arkouda that represent one object containing a single component. For example, the server side representation of a `pdarray`. When using `GenSymEntry` as the parent for these objects, all the properties are clearly defined its single component.

`GenSymEntry` can also be used when an element consists of multiple components, but the components are necessary for understanding a singular object. For example, `SegArraySymEntry` contains a `values` and a `segments` entry. Each of these are `pdarrays`, but `segments` is used to index into `values` and retrieve data. Here we have a very specific relationship between the components which ensures that the properties of `GenSymEntry` make sense.

Let's step through the `GenSymEntry` properties to get a sense of why objects like `SegArraySymEntry` inherit this class. 

- `dtype` refers to the data type of the `values` entry. All data points in `values` are required to be of that type. `segments` is always required to be an `int` array and is not considered data.
- `itemsize` again refers only to the size of a single data point in values. Additionally, because `segments` is always guaranteed to be of type `int`, we are still able to take `segments` into account when performning size computations.
- `size` refers to the number of data points in the `values` entry.
- `ndim` will store the number of segments because this is equivalent to our number of dimensions.
- `shape` currently stores the shape of the segments entry.

As you can see, despite the entry containing multiple `SymEntry` objects its components still fit the format provided by `GenSymEntry`.

<a id="parent-comp"></a>
### When to use `CompositeSymEntry`
Unlike `GenSymEntry`, `CompositeSymEntry` does not support objects containing only a single `SymEntry`. These objects are designed to handle objects containing multiple `SymEntry` objects that are not fully necessary to understand the object as whole. Additionally, these objects often have varying types of data between the `SymEntry` objects and unlike the `SegArraySymEntry` case, we do not know these types at compile time. Each `CompositeSymEntry` contains the properties below:

- `ndim` stores the number of dimensions in the entry. Depending on the object format this may represent different things.
- `size` stores the number of values in the entry. Again, this may represent different meaning depending upon the object.

<a id="unsure"></a>
### Still Unsure of which Parent Class to Use?
If you are unsure of which parent class to utilize when creating a new Symbol Table entry class, contact the core development team for Arkouda. You can either post an issue in our [GitHub](https://github.com/Bears-R-Us/arkouda/issues) or send a message on [Gitter](https://gitter.im/ArkoudaProject/community#).

<a id="build"></a>
## Building the New Symbol Table Entry
Once you have identified a parent class to inherit from, you are ready to configure the class representing the Symbol Table entry. Your new class should be added to `MultiTypeSymEntry.chpl`. An example class definition for each parent class is provided below. We will walk through defining each component separately.

**GenSymEntry**
```chapel
class NewSymEntry:GenSymEntry {
    // Properties specific to the class
    
    proc init() {
        super.init(elementType, objectSize);
        this.entryType = SymbolEntryType.EntryType;
        assignableTypes.add(this.entryType);
    }
    
    override proc getSizeEstimate(): int {
    
    }
}
```

**CompositeSymEntry**
```chapel
class NewSymEntry:CompositeSymEntry {
    // Properties specific to the class
    
    proc init() {
        super.init(size);
        this.entryType = SymbolEntryType.EntryType;
        assignableTypes.add(this.entryType);
    }
    
    override proc getSizeEstimate(): int {
    
    }
}
```

<a id="enum"></a>
### Add to `SymbolEntryType` Enum
The `SymbolEntryType` enum is used to keep track of Symbol Table entry types. When a entry is initialized, the enum value is stored to the `assignableTypes` property. This property is then used to determine valid Symbol Table entry types that an entry can be cast to. When initially accessing a Symbol Table entry, usually an `AbstractSymEntry` is returned. This class contains the `assignableTypes` property and allows for easy identification of the entry's valid types.

As an example, let's look at adding a `NewSymEntry` that inherits from `CompositeSymEntry` to the enum object. Notice indentation is used to indicate parent-child relationships. The result of adding `NewSymEntry` is below:

```chapel
enum SymbolEntryType {
    AbstractSymEntry,  // Root Type from which all other types will inherit
    
        TypedArraySymEntry, // Parent type for Arrays with a dtype, legacy->GenSymEntry
            PrimitiveTypedArraySymEntry, // int, uint8, bool, etc.
            ComplexTypedArraySymEntry,   // DateTime, TimeDelta, IP Address, etc.
    
        GenSymEntry,
            SegStringSymEntry,    // SegString composed of offset-int[], bytes->uint(8)
            CategoricalSymEntry,  // Categorical
            SegArraySymEntry,     // Segmented Array

        CompositeSymEntry,        // Entries that consist of multiple SymEntries of varying type
            GroupBySymEntry,      // GroupBy
            NewSymEntry,          // Your New Symbol Table Entry

        AnythingSymEntry, // Placeholder to stick aritrary things in the map
        UnknownSymEntry,
        None
}
```

<a id="props"></a>
### Adding Properties
In addition to its inherited properties, your new Symbol Table entry will need to define properties specific to itself. In most cases, these will be `SymEntry` objects contained within this entry. 

Let's look at an example where we have 2 `SymEntry` objects as properties. The first will be an array of strings and the other an array of integers. For our example, we will make the following assumptions:

- The `SymEntry` containing a strings array is our main data.
- The `SymEntry` containing an integer array contains the index of the midpoint of each string.

Now let's build out the class, starting with properties. Based on our assumptions, `GenSymEntry` is going to be our parent.

```chapel
class NewSymEntry:GenSymEntry {
    var stringsEntry: shared SymEntry(string);
    var midptEntry: shared SymEntry(int);
}
```

<a id="init"></a>
### Initializing the Object
Now that we have our properties defined, let's look at initializing the object. Because we need to set our 2 properties, we need to pass 2 `SymEntry` objects to our constructor. 

```chapel
class NewSymEntry:GenSymEntry {
    var stringsEntry: shared SymEntry(string);
    var midptEntry: shared SymEntry(int);
    
    proc init(stringsEntry: shared SymEntry, midptEntry: shared SymEntry){
        super.init(string, stringsEntry.size);
        this.entryType = SymbolEntryType.NewSymEntry;
        assignableTypes.add(this.entryType);
        this.etype = string;
        
        this.stringsEntry = stringsEntry;
        this.midptEntry = midptEntry;
        
        this.dtype = whichDtype(string);
        this.itemsize = this.stringsEntry.itemsize;
        this.size = this.midptEntry.size;
        this.ndim = this.midptEntry.ndim;
        this.shape = this.midptEntry.shape;
    }
}
```

This `init` does a lot so let's break it down. First, we are initializing the `GenSymEntry` components. Here we use the `stringsEntry.size` to set the size because that is the principle component and will set size equal to our number of strings. We use `string` for the type because our main component is of type `string`. This will also be the `etype` for the entry we are building. Next, we need to ensure that we can validate the type of entry, so we assign the entry type and add it to our `assignableTypes`. Remember, this is key to being able to access the object later. Next, we assign our `SymEntry` objects. Finally, we configure the components from the parent. This will differ for each Symbol Table Entry, but walking through why we made the decisions we did here should help in your decision making.

The first component is setting the dtype equivalent of the Chapel type. This is used to identify handling in situations where different types may be required. `itemsize` is set to `stringsEntry.itemsize` because this component is our main data source. Thus, the `itemsize` of our new entry should be equivalent. For the final three components, we could use the `size`, `ndim` and `shape` of either our `stringsEntry` or our `midptEntry`. The key for these properties is to ensure that they represent the object correctly.

<a id="sizeEst"></a>
### Providing the Size Estimation
The size estimation is used for memory management and to ensure that objects can be accessed without exceeding memory thresholds. In this example, we need the computation to account for the size of each of our components, namely `stringsEntry` and `midptEntry`. As a result, our computation is simply adding the two sizes together. See the updated class definition below.

```chapel
class NewSymEntry:GenSymEntry {
    var stringsEntry: shared SymEntry(string);
    var midptEntry: shared SymEntry(int);
    
    proc init(stringsEntry: shared SymEntry, midptEntry: shared SymEntry){
        super.init(string, stringsEntry.size);
        this.entryType = SymbolEntryType.NewSymEntry;
        assignableTypes.add(this.entryType);
        this.etype = string;
        
        this.stringsEntry = stringsEntry;
        this.midptEntry = midptEntry;
        
        this.dtype = whichDtype(string);
        this.itemsize = this.stringsEntry.itemsize;
        this.size = this.midptEntry.size;
        this.ndim = this.midptEntry.ndim;
        this.shape = this.midptEntry.shape;
    }
    
    override proc getSizeEstimate(): int {
        return this.stringsEntry.getSizeEstimate() + this.midptEntry.getSizeEstimate();
    }
}
```

Now we have a completed Symbol Table entry class. In the next section, we will look at one additional example to handle dynamic typing.

<a id="dynamic"></a>
### Handle Dynamic Types
What if your data can contain values of different types in the `SymEntry` objects it contains? No problem! We can assign the type during initialization. Let's look at `SegArraySymEntry` as an example. The relevant code is provided below.

```chapel
class SegArraySymEntry:GenSymEntry {
    type etype;

    var segmentsEntry: shared SymEntry(int);
    var valuesEntry: shared SymEntry(etype);

    proc init(segmentsSymEntry: shared SymEntry, valuesSymEntry: shared SymEntry, type etype){
        super.init(etype, valuesSymEntry.size);
        this.entryType = SymbolEntryType.SegArraySymEntry;
        assignableTypes.add(this.entryType);
        this.etype = etype;
        this.segmentsEntry = segmentsSymEntry;
        this.valuesEntry = valuesSymEntry;

        this.dtype = whichDtype(etype);
        this.itemsize = this.valuesEntry.itemsize;
        this.size = this.segmentsEntry.size;
        this.ndim = this.segmentsEntry.ndim;
        this.shape = this.segmentsEntry.shape;
    }

    override proc getSizeEstimate(): int {
        return this.segmentsEntry.getSizeEstimate() + this.valuesEntry.getSizeEstimate();
    }
}
```

Notice the line `type etype;`. This indicates to the class that we will be setting a property type during initialization of the object. We can see that this type will be used to define the `valuesEntry`. This indicates that the values entry will be a `SymEntry` of the type `etype` and that type will be set in the `init` process. 

Now look at the `proc init()`. The last parameter here is `type etype`. This requires us to pass the Chapel type into the `init` process. This type is then used to initialize the `GenSymEntry` to the proper type. Please note here that the `valuesSymEntry` must be of the type `etype` as well. 

The remainder of the class is essentially the same configuration as our previous example. 

<a id="create"></a>
## Creating the Object
Once you have your Symbol Table entry, you will need to configure the object that is built from the entry. This object will be used during processing on the Arkouda server. Let's look at creating this object that corresponds to our `NewSymEntry` example.

First, you will need to create a new Chapel file to house the module. We will call this `NewSymObject.chpl`. Then, we will define the module in this file. Please note - we will not be highlighting the imports in this tutorial, but you will most likely need to import other Chapel/Arkouda modules.

```chapel
module NewSymObject {

}
```

We will now define the class for the object. Let's discuss some of the properties. We need to store the `name` associated with the `SymEntry` used to create the object. The next attribute we need is `composite`, which contains our `NewSymEntry` object. We will store its component entries as well. These objects will need to be passed to our class constructor.

```chapel
module NewSymObject {
    class NewSymObj {
        var name: string;
    
        var composite: borrowed NewSymEntry;
    
        var midpts: shared SymEntry(int);
        var values: shared SymEntry(string);
        
        proc init(entryName:string, entry:borrowed NewSymEntry) {
            name = entryName;
            composite = entry;
            
            midpts = composite.midptEntry;
            values = composite.valuesEntry;
        }
    }
}
```

Now that we have our object, we need to add methods to create it from Chapel arrays. These methods will create the required `SymEntry` objects and use them to generate a `NewSymEntry` object. Let's update our module to add this process.

```chapel
module NewSymObject {
    proc getNewSymObj(midpts: [] int, values: [] string, st: borrowed SymTab): owned NewSymObj throws {
        var midptEntry = new shared SymEntry(midpts);
        var valuesEntry = new shared SymEntry(values);
        var newEntry = new shared NewSymEntry(midptEntry, valuesEntry);
        var name = st.nextName();
        st.addEntry(name, newEntry);
        return getNewSymObj(name, st, newEntry.etype);
    }
    
    class NewSymObj {
        var name: string;
    
        var composite: borrowed NewSymEntry;
    
        var midpts: shared SymEntry(int);
        var values: shared SymEntry(string);
        
        proc init(entryName:string, entry:borrowed NewSymEntry) {
            name = entryName;
            composite = entry;
            
            midpts = composite.midptEntry;
            values = composite.valuesEntry;
        }
    }
}
```

Notice that we have a call to `getNewSymObj()`. This overload call allows us to build the `NewSymObj` from the name of the `NewSymEntry`. This is important because it allows us to access the borrowed object from the name. This will be important when we add messages to perform various operations on an already created object because it allows easy access of the object. Let's now build this overload function in our module.

```chapel
module NewSymObject {
    proc getNewSymObj(name: string, st: borrowed SymTab): owned NewSymObj throws {
        var abstractEntry = st.lookup(name);
        if !abstractEntry.isAssignableTo(SymbolEntryType.NewSymEntry) {
            var errorMsg = "Error: Unhandled SymbolEntryType %s".format(abstractEntry.entryType);
            saLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw new Error(errorMsg);
        }
        var entry: NewSymEntry = abstractEntry: borrowed NewSymEntry();
        return new owned NewSymObj(name, entry);
    }
    proc getNewSymObj(midpts: [] int, values: [] string, st: borrowed SymTab): owned NewSymObj throws {
        var midptEntry = new shared SymEntry(midpts);
        var valuesEntry = new shared SymEntry(values);
        var newEntry = new shared NewSymEntry(midptEntry, valuesEntry);
        var name = st.nextName();
        st.addEntry(name, newEntry);
        return getNewSymObj(name, st, newEntry.etype);
    }
    
    class NewSymObj {
        var name: string;
    
        var composite: borrowed NewSymEntry;
    
        var midpts: shared SymEntry(int);
        var values: shared SymEntry(string);
        
        proc init(entryName:string, entry:borrowed NewSymEntry) {
            name = entryName;
            composite = entry;
            
            midpts = composite.midptEntry;
            values = composite.valuesEntry;
        }
    }
}
```

We now have a completed module. Obviously, additional functionality can be added as desired. It is also of note that in order to create the `NewSymEntry` you will need to construct a server message to create the object. For more details to get started with that, take a look at Arkouda's [MESSAGING_OVERVIEW.md](https://github.com/Bears-R-Us/arkouda/blob/master/training/MESSAGING_OVERVIEW.md).