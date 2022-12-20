.. _examples-label:

*************
Examples
*************

Arkouda Arrays
====================

Arkouda arrays function similarly to arrays in NumPy, but allow for a much larger scale. In Arkouda, arrays are referred to as `pdarray` objects. It is possible to generate a `pdarray` from a Python `list`, NumPy `ndarray`, or using a generator method similar to those found in NumPy. This document aims to provide an introduction into some of the most commonly used elements of Arkouda and is not an exhaustive description of functionality.

`pdarray` Creation
-------------------

Like `ndarray` objects in NumPy, Arkouda `pdarray` objects can be generated from a Python `list`.

.. code-block:: python

    # create the Python List
    >> l = [0, 1, 2, 3, 4]
    
    # generate a pdarray
    >> ak_arr = ak.array(l)
    >> ak_arr
    array([0 1 2 3 4])

`pdarray` objects can be generated directly from an `ndarray`. This allows you to easily move objects into Arkouda from NumPy.

.. code-block:: python

    # create an ndarray
    >> np_arr = np.array([0, 1, 2, 3, 4])

    # generate a pdarray
    >> ak_arr = ak.array(np_arr)
    >> ak_arr
    array([0 1 2 3 4])

`pdarray` objects can be generated using generator calls such as `arange` and `randint`.

.. code-block:: python

    # arange
    >> ak_arr = ak.arange(10)
    >> ak_arr
    array([0 1 2 3 4 5 6 7 8 9])

    # randint(low, high, size)
    >> r = ak.randint(0, 100, 10)
    >> r # output will vary
    array([52 84 1 52 80 71 27 20 7 7])

Exporting `pdarray` Objects
---------------------------

Arkouda allows users to export `pdarray` objects to other formats to aide in transitioning between toolsets. A `pdarray` can be exported to a NumPy `ndarray` or a Python `list`.

.. code-block:: python

    # create pdarray
    >> ak_arr = ak.array([0, 1, 2, 3, 4])

    # export to ndarray
    >> np_arr = ak_arr.to_ndarray()
    >> np_arr
    array([0, 1, 2, 3, 4])

    # export to a Python List
    >> l = ak_arr.to_list()
    >> l
    [0, 1, 2, 3, 4]

`pdarray` Set operations
------------------------

Like NumPy, Arkouda supports set operations on `pdarray` objects. The supported set operations are 

- **IN** (`in1d`) : Test whether each element of a 1-D array is also present in a second array.
- **UNION** (`union1d`) : Compute the unique union of the arrays
- **INTERSECT** (`intersect1d`) : Compute the unique intersection of the arrays.
- **SET DIFFERENCE** (`setdiff1d`) : Compute the difference between the two arrays.
- **SYMMETRIC DIFFERENCE** (`setxor1d`) : Compute the exclusive-or of the two arrays.

One important note is that Arkouda takes this functionality beyond a single dimension. These operations can be performed on lists of `pdarrays` as well. We will look at `in1d` and `intersect1d` in both 1 dimension and multiple in the code block below.

.. code-block:: python

    # configure 2 pdarrays to run against
    >> a = ak.array([4, 2, 5, 6, 4, 7, 2])
    >> b = ak.array([1, 5, 4, 11, 9, 6])

    # compute boolean array indicating the values from a found in b.
    >> ak_in1d = ak.in1d(a, b)
    >> ak_in1d
    array([True False True True True False False])

    # compute array of unique values found in a and b
    >> ak_int = ak.intersect1d(a, b)
    >> ak_int
    array([4 5 6])

    # Arkouda can perform this operation on multiple arrays at once
    >> m1 =[
        ak.array([0, 1, 3, 4, 8, 5, 0]),
        ak.array([0, 9, 5, 1, 8, 5, 0])
    ]
    >> m2 =[
        ak.array([0, 1, 3, 4, 8, 7]),
        ak.array([0, 2, 5, 9, 8, 5])
    ]

    
    >> ak_in1dmult = ak.in1d(m1, m2)
    >> ak_in1dmulti
    array([True False True False True False True])
    
    >> ak_intmult = ak.intersect1d(m1, m2)
    >> ak_intmult
    [array([0 3 8]), array([0 5 8])]

There are a few things to keep in mind when working in the multi-dimension case. First, `m1` and `m2` must be Python `lists` containing the same number of `pdarray` elements. Second, the values are treated as a tuple. Using our example above, the first value of `m1` is viewed as `(0, 0)` during computation.

Arkouda DataFrames
====================

Like in Pandas, Arkouda supports the construct of a `DataFrame`. The structure of these objects is very similar, though some functionality may vary. `DataFrames` are extremely useful when working with multiple `pdarray` objects that are related. In Arkouda, `DataFrames` consist of an `Index` (which uses are `Arkouda.Index`), `Column Names` and `Column Data`.

Creating & Using a DataFrame
-----------------------------

Let's take a look at creating a `DataFrame` in Arkouda. Once again, we have several methods to create a `DataFrame` in Arkouda:

- Importing a Pandas `DataFrame`
- Python Mapping `{column_name: column_data}`. `column_data` must be `pdarray`. `column_name` will be used by the constructor to set the column names for the `DataFrame`

The most important thing to remember is that each column of an Arkouda `DataFrame` is a `pdarray` and must be provided as such. The only exception is when a Pandas DataFrame is being imported because the constructor will generate the `pdarray` objects for you from the columns of the Pandas `DataFrame`. 

Importing Pandas DataFrame
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    # construct the Pandas DataFrame
    >> fname = ['John', 'Jane', 'John', 'Jake']
    >> lname = ['Doe', 'Doe', 'Smith', 'Brown']
    >> age = [37, 35, 50, 32]
    >> salary = [75000, 77000, 100000, 35000]
    >> pd_df = pd.DataFrame({
        'F_Name': fname,
        'L_Name': lname,
        'Age': age,
        'Salary': salary
    })
    >> pd_df
        F_Name L_Name  Age  Salary
    0   John    Doe   37   75000
    1   Jane    Doe   35   77000
    2   John  Smith   50  100000
    3   Jake  Brown   32   35000

    # call the Arkouda DataFrame constructor
    >> df = ak.DataFrame(pd_df)
    >> df
        F_Name L_Name  Age  Salary
    0   John    Doe   37   75000
    1   Jane    Doe   35   77000
    2   John  Smith   50  100000
    3   Jake  Brown   32   35000 (4 rows x 4 columns)

Python Mapping
^^^^^^^^^^^^^^^

.. code-block:: python

    >> fname = ak.array(['John', 'Jane', 'John', 'Jake'])
    >> lname = ak.array(['Doe', 'Doe', 'Smith', 'Brown'])
    >> age = ak.array([37, 35, 50, 32])
    >> salary = ak.array([75000, 77000, 100000, 35000])
    >> df = ak.DataFrame({
        'F_Name': fname,
        'L_Name': lname,
        'Age': age,
        'Salary': salary
    })

    >> df
        F_Name L_Name  Age  Salary
    0   John    Doe   37   75000
    1   Jane    Doe   35   77000
    2   John  Smith   50  100000
    3   Jake  Brown   32   35000 (4 rows x 4 columns)

**NOTICE**: Here the call to the Arkouda `DataFrame` constructor takes in very close to the same information as the Pandas constructor, but with one key difference. Each of the columns is an Arkouda `pdarray`.

Basic Interaction
^^^^^^^^^^^^^^^^^

**Please Note:** For this section we will be using the same `DataFrame` generated in the creation demos.

In this section, we will highlight some of the basics of `DataFrame` interaction in Arkouda. You should notice that it is very similar to interacting with a Pandas `DataFrame`.

.. code-block:: python

    # adding reference to dataframe created earlier for easy reference
    >> df
        F_Name L_Name  Age  Salary
    0   John    Doe   37   75000
    1   Jane    Doe   35   77000
    2   John  Smith   50  100000
    3   Jake  Brown   32   35000 (4 rows x 4 columns)

    # accessing a column
    >> df['Age']
    array([37 35 50 32])

    # accessing multiple columns at once
    >> df['L_Name', 'Age'] # equivalent to df[['L_Name', 'Age']]
        L_Name  Age
    0    Doe   37
    1    Doe   35
    2  Smith   50
    3  Brown   32 (4 rows x 2 columns)

    # accessing row
    >> df[0]
    {'F_Name': 'John', 'L_Name': 'Doe', 'Age': 37, 'Salary': 75000}

    # accessing row slice
    >> df[0:2]
        F_Name L_Name  Age  Salary
    0   John    Doe   37   75000
    1   Jane    Doe   35   77000 (2 rows x 4 columns)

    # accessing multiple indexes
    >> idx = ak.array([0, 2, 3])
    >> df[idx]
        F_Name L_Name  Age  Salary
    0   John    Doe   37   75000
    2   John  Smith   50  100000
    3   Jake  Brown   32   35000 (3 rows x 4 columns)

Exporting to Pandas
--------------------

Exporting an Arkouda `DataFrame` to Pandas is extremely simple using the `to_pandas` function. 

.. code-block:: python

    # adding reference to dataframe created earlier for easy reference
    >> df
        F_Name L_Name  Age  Salary
    0   John    Doe   37   75000
    1   Jane    Doe   35   77000
    2   John  Smith   50  100000
    3   Jake  Brown   32   35000 (4 rows x 4 columns)

    >> pd_df = df.to_pandas()
    >> pd_df
        F_Name L_Name  Age  Salary
    0   John    Doe   37   75000
    1   Jane    Doe   35   77000
    2   John  Smith   50  100000
    3   Jake  Brown   32   35000

GroupBy
====================

In Pandas, groupby-aggregate is a very useful pattern that can be computationally intensive. Arkouda supports grouping by key and most aggregations in Pandas. `GroupBy` functionality in Arkouda is supported on `pdarray` and `DataFrame` objects.

`pdarrays`
-----------

.. code-block:: python

    # using randint for more interesting results. Note values will vary
    >> x = ak.randint(0, 10, 100)
    >> g = ak.GroupBy(x)
    >> g.count()
    (array([0 1 2 3 4 5 6 7 8 9]), array([14 5 8 17 14 8 5 9 11 9]))

DataFrames
-----------

.. code-block:: python

    # adding reference to dataframe created earlier for easy reference
    >> df
        F_Name L_Name  Age  Salary
    0   John    Doe   37   75000
    1   Jane    Doe   35   77000
    2   John  Smith   50  100000
    3   Jake  Brown   32   35000 (4 rows x 4 columns)

    >> g = df.groupby("L_Name")
    >> g.count()
    Doe      2
    Brown    1
    Smith    1
    dtype: int64

