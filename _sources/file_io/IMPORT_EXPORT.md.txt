# Import/Export

Arkouda allows for importing and exporting data in Pandas format, specifically DataFrames. This functionality is currently performed on the client. As a result it is assumed that the size of data being imported can be handled by the client because it was written by Pandas. Arkouda natively verifies that the size of data being sent to client can be handled.

During both import and export operations, file type is maintained. Thus, if you import/export an HDf5 file and elect to save an appropriately formatted file during the operation, the resulting file will also be HDF5.

This functionality should not be required for Parquet files, but is supported for both HDF5 and Parquet.

## Export

Export takes a file that was saved using Arkouda and reads it into Pandas. The user is able to specify if they would like to save the result to a file that can be read by Pandas and/or return the resulting Pandas object.

## Import

Importing data takes a file that was saved using Pandas and reads it into Arkouda. The user is able to specify if they would like to save the result to a file that can be read by Arkouda and/or return the resulting Arkouda object.

## API Reference

```{eval-rst}
- :py:func:`arkouda.io.import_data`
- :py:func:`arkouda.io.export`
```
