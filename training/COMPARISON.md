# Arkouda vs NumPy/Pandas
This document compares and contrasts Arkouda with `NumPy` and `Pandas`. As you go through this document, we encourage you to try the code for yourself &mdash; especially in the final section.

<a id="toc"></a>
## Table of Contents
1. [Background Information](#bg)
2. [Importing Arkouda](#import)
3. [Disconnecting From Arkouda Server](#discon)
4. [Arrays](#arrays)
   1. [Creating Arrays in NumPy](#nparray_create)
   2. [Creating Arrays in Arkouda](#akarray_create)
   3. [Indexing](#ind)
   4. [Set Operations](#setops)
   5. [GroupBy](#groupby)
   6. [Not Supported](#no_support)
5. [Dataframes](#df)
   1. [Creating Pandas DataFrame](#df_pd_create)
   2. [Creating Arkouda DataFrame](df_ak_create)
   3. [GroupBy on DataFrame](#df_groupby)
6. [Example Application](#ex)

<a id="bg"></a>
## Background Information

Arkouda began with the intention of allowing `NumPy`-like functionality in an HPC setting. Over time, it has evolved to support `Pandas` functionality and continues to expand. The API mimics `NumPy` and `Pandas` to allow an easy transition for users familiar with these Python packages.

So what makes Arkouda special? The answer is simple, scalability. Where other packages run against compiled C/C++ code, Arkouda leverages Chapel. This allows Arkouda to run locally and gain significant performance benefits in an HPC environment. 

<a id="import"></a>
## Importing Arkouda
This may seem like trivial step to highlight. However, Arkouda requires the user to connect to the server component before running commands.

### Local Connection
When running the Arkouda server on the same machine as your python code, no parameters are required to connect. This uses the default settings of `server=localhost` and `port=5555`. During development this is typically what is used.
```python
import arkouda as ak
ak.connect()
```

### Custom Connection
Arkouda allows for custom connections as well. This is useful for connecting to remote servers. More information on connection parameters can be found [here](https://bears-r-us.github.io/arkouda/autoapi/arkouda/client/index.html?highlight=connect#arkouda.client.connect).

<a id="discon"></a>
## Disconnecting From Arkouda Server
Because you are connected to the Arkouda server, when you are done working you will need to disconnect from the server. There are 2 ways to do this:

- `ak.disconnect()` - This will only disconnect the client from the server, but the server will remain running. 
- `ak.shutdown()` - This will disconnect from the server AND shutdown the server.

<a id="arrays"></a>
## Arrays
Arkouda arrays, referred to as `pdarrays`, are at the base of all Arkouda functionality. These are very similar to `NumPy` arrays. 

*It is important to note that arrays are immutable objects due to the configurations. However, there are functions that allow for updates to the array artificially. These functions will generate an entirely new array.*

<a id="nparray_create"></a>
### Creating Arrays in Numpy
There are several ways to construct arrays using `NumPy`. The first is using a Python List. 
```python
import numpy as np
arr = np.array([0, 1, 2, 3, 4])
arr
array([0, 1, 2, 3, 4])
```

`NumPy` arrays can also be generated using `np.arange()`. This is useful for creating ranges of a provided size.
```python
import numpy as np
arr = np.arange(5)
arr
array([0, 1, 2, 3, 4])
```

<a id="akarray_create"></a>
### Creating Arrays in Arkouda
Arkouda array creation is very similar to `NumPy`. Arrays can also be created using Python Lists.
```python
import arkouda as ak
ak.connect()
arr = ak.array([0, 1, 2, 3, 4])
arr
array([0 1 2 3 4])
```

Arkouda also supports generation of arrays from `NumPy` arrays.
```python
import arkouda as ak
import numpy as np 
ak.connect()
nparr = np.array([0, 1, 2, 3, 4])
arr = ak.array(nparr)
arr
array([0 1 2 3 4])
```

If you need to generate a range of a given size, use `ak.arange()`.
```python
import arkouda as ak
ak.connect()
arr = ak.arange(5)
arr
array([0 1 2 3 4])
```

<a id="ind"></a>
### Indexing
Arkouda and `Numpy` support a wide range of indexing. The API here is virtually identical except for array indexing due to Arkouda requiring a `pdarray`. Indexing formats supported are `integer`, `slice`, `integer array`, and `boolean array`.

*NOTE: For `boolean` indexing the boolean array provided must be the same size as the array being indexed.*

#### NumPy Indexing
```python
import numpy as np
np_a = np.arange(10)
# Integer Indexing
np_a[5]
5

# Slice Indexing 
np_a[0:4:2]
array([0, 2])

# Array Indexing
np_a[[0, 3, 4, 5, 9]]
array([0, 3, 4, 5, 9])

# Boolean Indexing
np_a[[True, False, False, True, True, True, False, True, False, True]]
array([0, 3, 4, 5, 7, 9])
```

#### Arkouda Indexing
```python
import arkouda as ak
ak_a = ak.arange(10)
# Integer Indexing
ak_a[5]
5

# Slice Indexing 
ak_a[0:4:2]
array([0, 2])

# Array Indexing
ak_a[ak.array([0, 3, 4, 5, 9])]
array([0, 3, 4, 5, 9])

# Boolean Indexing
ak_a[ak.array([True, False, False, True, True, True, False, True, False, True])]
array([0, 3, 4, 5, 7, 9])
```


<a id="setops"></a>
### Set Operations
Arkouda and `NumPy` support set operations. Just like array creation, these functions are very similar. Both packages support `in1d`, `intersect1d`, `union1d`, `setdiff1d`, and `setxor1d`. In this section we will only demonstrate utilization of `intersect1d`. Documentation on all set operations in Arkouda can be found below:

- [in1d](https://bears-r-us.github.io/arkouda/autoapi/arkouda/pdarraysetops/index.html?highlight=in1d#arkouda.numpy.pdarraysetops.in1d)
- [intersect1d](https://bears-r-us.github.io/arkouda/autoapi/arkouda/pdarraysetops/index.html?highlight=intersect1d#arkouda.numpy.pdarraysetops.intersect1d)
- [union1d](https://bears-r-us.github.io/arkouda/autoapi/arkouda/pdarraysetops/index.html?highlight=union1d#arkouda.numpy.pdarraysetops.union1d)
- [setdiff1d](https://bears-r-us.github.io/arkouda/autoapi/arkouda/pdarraysetops/index.html?highlight=setdiff1d#arkouda.numpy.pdarraysetops.setdiff1d)
- [setxor1d](https://bears-r-us.github.io/arkouda/autoapi/arkouda/pdarraysetops/index.html?highlight=setdiff1d#arkouda.numpy.pdarraysetops.setxor1d)

#### Intersect1d Arkouda vs NumPy
```python
# NumPy
import numpy as np
np_arr = np.array([4, 2, 5, 6, 4, 7, 2])
np_arr2 = np.array([1, 5, 4, 11, 9, 6])
np_int = np.intersect1d(np_arr, np_arr2)
np_int
array([4, 5, 6])

# Arkouda
import arkouda as ak
ak.connect()
ak_arr = ak.array(np_arr)
ak_arr2 = ak.array(np_arr2)
ak_int = ak.intersect1d(ak_arr, ak_arr2)
ak_int
array([4 5 6])
```

Arkouda takes this one step further and allows set operations on sequences of `pdarray` objects. This requires all `pdarrays` in the sequence to be the same size. 

```python
import arkouda as ak
ak.connect()
m1 =[
    ak.array([0, 1, 3, 4, 8, 5, 0]),
    ak.array([0, 9, 5, 1, 8, 5, 0])
]
m2 =[
    ak.array([0, 1, 3, 4, 8, 7]),
    ak.array([0, 2, 5, 9, 8, 5])
]
ak_int = ak.intersect1d(m1, m2)
ak_int
[array([0 3 8]), array([0 5 8])]
```

<a id="groupby"></a>
### GroupBy
Both Arkouda and `NumPy` support the idea of a GroupBy. However, this is an area where the APIs differ significantly. If you are familiar with SQL `GROUP BY` statements, the concept here is very similar.

#### GroupBy in NumPy
While the concept is the same as Arkouda's `GroupBy`, `NumPy` does not explicitly support `GroupBy`. Instead, it uses `np.unique`.
```python
import numpy as np
arr = np.array([4, 2, 5, 6, 4, 7, 2])
keys, cts = np.unique(arr, return_counts=True)
keys  # this represents the unique keys in the array
array([2, 4, 5, 6, 7])
cts  # this represents the number of times each key occurs in the array
array([2, 2, 1, 1, 1])
```

Arkouda's `GroupBy` is a bit more advanced. Arkouda has a `GroupBy` class that leverages functionality similar to `np.unique` to compute its components. This class also includes `.permutation` and `.segment` properties. `.permutation` contains a pdarray whose values are the indexes of the original array in order by group. `.segments` contains a pdarray whose values are the starting index of each group (when `GroupBy.keys` is indexed by `GroupBy.permutation`).
```python
import arkouda as ak
ak.connect()
arr = ak.array([4, 2, 5, 6, 4, 7, 2])
g = ak.GroupBy(arr)
keys, cts = g.count()

keys  # this represents the unique keys in the array
array([2 4 5 6 7])
cts  # this represents the number of times each key occurs in the array
array([2 2 1 1 1])

# the unique keys can also be accessed via the .unique_keys property
g.unique_keys
array([2 4 5 6 7])

g.permutation
array([1 6 0 4 2 3 5])
g.segments
array([0 2 4 5 6])
g.keys[g.permutation]
array([2 2 4 4 5 6 7])
```

Arkouda is able to create `GroupBy` objects from a sequence of groupable objects. 
```python
import arkouda as ak
ak.connect()
a = ak.array([0, 1, 2, 3, 3, 3, 2, 1])
b = ak.array([0, 1, 1, 2, 3, 3, 1, 1])
g = ak.GroupBy([a, b])
keys, cts = g.count()

keys  # this represents the unique keys in the array
(array([0 1 2 3 3]), array([0 1 1 2 3]))  

cts  # this represents the number of times each key occurs in the array
array([1 2 2 1 2])

g.permutation
array([0 1 7 2 6 3 4 5])
g.segments
array([0 1 3 5 6])
g.keys[0][g.permutation]
array([0 1 1 2 2 3 3 3])
g.keys[1][g.permutation]
array([0 1 1 1 1 2 3 3])
```
Notice that keys is a tuple of length equal to the number of elements grouped on. The corresponding indexes are the keys. For example, key 0 is `(0, 0)`.

### Not Supported
A common thing to do in Python is iterate over an object. This is not something that should be done (and is not directly supported) with Arkouda arrays.

For example, attempting to iterate directly on a `pdarray` will raise and exception.
```python
import arkouda as ak
ak.connect()
array = ak.arange(100)
for i in array:
    print(i)

NotImplementedError: pdarray does not support iteration. To force data transfer from server, use to_ndarray
```

Unfortunately, there is a way to iterate `pdarray` objects without raising an exception. This is for informational purposes only. If you find yourself writing code similar to what appears below or wanting to iterate a `pdarray`, there's likely a way to get the desired behavior with an array oriented method in arkouda or the functionality will need to be implemented in Chapel.

```python
import arkouda as ak
ak.connect()
array = ak.arange(100)
for i in array.to_list():
    print(i)
```

```python
import arkouda as ak
ak.connect()
array = ak.arange(100)
for i in range(array.size):
    print(array[i])
```

**NOTE: The above 2 examples will run. However, it is extremely inefficient and should NEVER be utilized.**

If you find yourself in one of the situations demonstrated above, please contact the Arkouda development team for assistance. It is likely that there is another way to handle what you need. If there is not, it can be done in Chapel and the team will assist with that as well.

<a id="df"></a>
## DataFrames
Arkouda and `Pandas` support `DataFrame` objects. Throughout this section, you will notice that there are some key differences. Arkouda DataFrame support is fairly new and is continually being updated. The most common way to construct a `DataFrame` in both packages is by using a dictionary. However, other methods exist. Links to the documentation are below if you would like to review alternative methods. For the purposes of this document, we will be using Python Dictionaries.

- [Pandas DataFrame Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
- [Arkouda DataFrame Documentation](https://bears-r-us.github.io/arkouda/usage/dataframe.html?highlight=dataframe#arkouda.DataFrame)

<a id="df_pd_create"></a>
### Creating Pandas DataFrame
Pandas can generate a DataFrame from data where each column is a Python List or a `NumPy` array. The dictionary structure is demonstrated below:

```json
{
    "column_name_1": [0, 1, 2],
    "column_name_2": ["a", "b", "c"]
}
```

```python
import pandas as pd
pd_df = pd.DataFrame({
    'F_Name': ['John', 'Jane', 'John', 'Jake'],
    'L_Name': ['Doe', 'Doe', 'Smith', 'FromStateFarm'],
    'Age': [37, 35, 50, 32],
    'Salary': [75000, 77000, 100000, 35000]
})
pd_df
  F_Name         L_Name  Age  Salary
0   John            Doe   37   75000
1   Jane            Doe   35   77000
2   John          Smith   50  100000
3   Jake  FromStateFarm   32   35000
```

<a id="df_ak_create"></a>
### Creating Arkouda DataFrame
Arkouda DataFrames require the columns to be `pdarray` objects. The dictionary structure is below:
```json
{
   "column_name_1": ak.array([0, 1, 2]),
   "column_name_2": ak.array(["a", "b", "c"])
}
```

```python
import arkouda as ak
ak.connect()
ak_df = ak.DataFrame({
    'F_Name': ak.array(['John', 'Jane', 'John', 'Jake']),
    'L_Name': ak.array(['Doe', 'Doe', 'Smith', 'FromStateFarm']),
    'Age': ak.array([37, 35, 50, 32]),
    'Salary': ak.array([75000, 77000, 100000, 35000])
})
ak_df
  F_Name         L_Name  Age  Salary
0   John            Doe   37   75000
1   Jane            Doe   35   77000
2   John          Smith   50  100000
3   Jake  FromStateFarm   32   35000 (4 rows x 4 columns)
```

<a id="df_groupby"></a>
### GroupBy on DataFrames
`GroupBy` on DataFrames is similar in concept to `GroupBy` on arrays. When calling `.count()` on the resulting `GroupBy` object there is some variation in the return from Arkouda and `Pandas`. 

#### GroupBy on Pandas DataFrame
One important note is that Pandas supports grouping on both axes. Arkouda does not currently support this.
```python
import pandas as pd
pd_df = pd.DataFrame({
    'F_Name': ['John', 'Jane', 'John', 'Jake'],
    'L_Name': ['Doe', 'Doe', 'Smith', 'FromStateFarm'],
    'Age': [37, 35, 50, 32],
    'Salary': [75000, 77000, 100000, 35000]
})
pd_g = pd_df.groupby("F_Name")
pd_g.count()
        L_Name  Age  Salary
F_Name                     
Jake         1    1       1
Jane         1    1       1
John         2    2       2
```
Notice that the return of count is a `DataFrame` object with each value representing the number of different values in a given column for the corresponding grouped column.

#### GroupBy on Arkouda DataFrame
Running `GroupBy` on an Arkouda `DataFrame` results in an Arkouda `GroupBy` object. This results in different output when running `.count()` on the result.
```python
import arkouda as ak
ak.connect()
ak_df = ak.DataFrame({
    'F_Name': ak.array(['John', 'Jane', 'John', 'Jake']),
    'L_Name': ak.array(['Doe', 'Doe', 'Smith', 'FromStateFarm']),
    'Age': ak.array([37, 35, 50, 32]),
    'Salary': ak.array([75000, 77000, 100000, 35000])
})
ak_g = ak_df.GroupBy("F_Name")
ak_g.count()
(array(['John', 'Jane', 'Jake']), array([2 1 1]))
```
Notice the return here is the unique keys found in the column requested to group on along with the number of time each of those keys occurred in the column. 

<a id="ex"></a>
## Example Application
Now that we have highlighted some of Arkouda's key components, let's walk through an example of how to use them. A good example of this is walking through the code for a set operation on a sequence of `pdarray` objects. We will go through one of these functions, `setdiff1d`. The goal here is to identify the values only found in the first object provided.

In case you are unfamiliar with `setdiff1d` (set difference), the goal is to compute the the difference between 2 sets. A brief example is shown below.

```python
# given set 1
s1 = [1,2,3,4]
s2 = [3,7,8,2]

# set difference computes the values in s1 that are not found in s2
s1 - s2 = [1, 4]
```

*NOTE: You will be using the `ak.concatenate()` function in these examples. This function creates a new array containing all values from the provided `pdarray`s.*

For the example the following assumptions will be made:
- The provided arrays are not unique
- We will not be validating the format of the input as it is specifically designed for these examples
- Input will be of the format below. Referenced as `m1` and `m2` in the examples.

First, import and connect to Arkouda and initialize `m1` and `m2`.
```python
import arkouda as ak
ak.connect()

m1 =[
    ak.array([0, 1, 3, 4, 8, 5, 0]),
    ak.array([0, 9, 5, 1, 8, 5, 0])
]
m2 =[
    ak.array([0, 1, 3, 4, 8, 7]),
    ak.array([0, 2, 5, 9, 8, 5])
]
```
Next, we will create a key to de-interleave the arrays. Using this to index into the concatenated arrays will return values from `m1`. We will also concatenate `m1` and `m2`. `ak.ones()` creates a `pdarray` with every value being `1`. `ak.zeros()` creates a `pdarray` with every value being `0`. 
```python
# Key for deinterleaving result
isa = ak.concatenate(
   (ak.ones(m1[0].size, dtype=ak.bool), ak.zeros(m2[0].size, dtype=ak.bool)), ordered=False
)
# for instructional purposes, displaying the contents of isa (more details below)
isa
array([True True True True True True True False False False False False False])


# concatenate m1 and m2
c = [ak.concatenate(x, ordered=False) for x in zip(m1, m2)]

#displaying c for clarity. More details below
c
[array([0 1 3 4 8 5 0 0 1 3 4 8 7]), array([0 9 5 1 8 5 0 0 2 5 9 8 5])]
```

After this step, `isa` is a pdarray with `True` in the indexes where values are from `m1` and `False` in the indexes where values are from `m2`. This is because we cast the values from `1` and `0` to their equivalent boolean values using the `dtype=ak.bool` parameter. `c` is a sequence of 2 arrays. Notice that `c[0]` is equal to `[m1[0], m2[0]]` and `c[1]` is equal to `[m1[1], m2[1]]`. `c` is the result of concatenating the keys resulting in a sequence of equal length to `m1` and `m2` and where each value in the sequence is of `size[i] = m1[i].size + m2[i].size`. It is important to note that these values may not always be in an obvious order. This is because we are using `ordered=False` which allows the system to return the concatenation in the order that makes the most sense. Here, we are running locally, which makes the order appear as expected. However, on a distributed system, the values corresponding to `m1` and `m2` may be interleaved, resulting in the need for this computation.

Now, we need to create a `GroupBy` object to get our unique keys. And the counts of those keys.
```python
g = ak.GroupBy(c)
k, ct = g.count()
k
(array([0 1 1 3 4 4 5 7 8]), array([0 2 9 5 1 9 5 5 8]))
ct
array([3 1 1 2 1 1 1 1 2])
```

This next step does a lot for us, so we are going to get in more detail here. 
```python
truth = g.broadcast(ct == 1, permute=True)

#displaying for context
truth
array([False True False True False True False False True False True False True])

```
We need to compute the unique keys that appear in the union only once; `ct == 1` provides that information. We then broadcast the resulting `True`/`False` value to the original keys the `GroupBy` was built from. Setting `permute=True` ensures that resulting boolean array is in the same order as the original keys, which is crucial to the next step. As a result, `truth` provides us with a boolean index to the keys that are found only once in the union of `m1` and `m2`. 

Next, we will access the `truth` array only at indexes corresponding to `m1` because we wish to return the keys from `m1` that are not present in `m2`. Since `truth` is the boolean index to unique keys in the union, we will use `isa` to return on the values from `m1`. *Reminder - isa is the boolean index for values in `m1` in the concatenated keys.*
```python
rtnIndx = truth[isa]
```

Finally, we iterate the sequence of `pdarray` objects in `m1` and index each one by the boolean array `rtnIndx`. This will give us the desired return of values in `m1` that are not in `m2`.

```python
result = [x[rtnIndx] for x in m1]
result
[array([1 4 5]), array([9 1 5])]
```

For more examples, visit the [ArkoudaNotebooks](https://github.com/Bears-R-Us/ArkoudaNotebooks) repository.