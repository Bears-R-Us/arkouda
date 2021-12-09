#include "ArrowFunctions.h"

#include <iostream>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/column_reader.h>

/*
  Arrow Error Helpers
  -------------------
  Arrow provides PARQUETASSIGNORTHROW and other similar macros
  to help with error handling, but since we are doing something
  unique (passing back the error message to Chapel to be displayed),
  these helpers are similar to the provided macros but matching our
  functionality. 
*/

// The `ARROWRESULT_OK` macro should be used when trying to
// assign the result of an Arrow/Parquet function to a value that can
// potentially throw an error, so the argument `cmd` is the Arrow
// command to execute and `res` is the desired variable to store the
// result
#define ARROWRESULT_OK(cmd, res)                                \
  {                                                             \
    auto result = cmd;                                          \
    if(!result.ok()) {                                          \
      *errMsg = strdup(result.status().message().c_str());      \
      return ARROWERROR;                                        \
    }                                                           \
    res = result.ValueOrDie();                                  \
  }

// The `ARROWSTATUS_OK` macro should be used when calling an
// Arrow/Parquet function that returns a status. The `cmd`
// argument should be the Arrow function to execute.
#define ARROWSTATUS_OK(cmd)                     \
  if(!check_status_ok(cmd, errMsg))             \
    return ARROWERROR;

bool check_status_ok(arrow::Status status, char** errMsg) {
  if(!status.ok()) {
    *errMsg = strdup(status.message().c_str());
    return false;
  }
  return true;
}

/*
 C++ functions
 -------------
 These C++ functions are used to call into the Arrow library
 and are then called to by their corresponding C functions to
 allow interoperability with Chapel. This means that all of the
 C++ functions must return types that are C compatible.
*/

int cpp_getNumRows(const char* filename, char** errMsg) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  ARROWRESULT_OK(arrow::io::ReadableFile::Open(filename, arrow::default_memory_pool()),
                 infile);

  std::unique_ptr<parquet::arrow::FileReader> reader;
  ARROWSTATUS_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
  
  return reader -> parquet_reader() -> metadata() -> num_rows();
}

int cpp_getType(const char* filename, const char* colname, char** errMsg) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  ARROWRESULT_OK(arrow::io::ReadableFile::Open(filename, arrow::default_memory_pool()),
                 infile);

  std::unique_ptr<parquet::arrow::FileReader> reader;
  ARROWSTATUS_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

  std::shared_ptr<arrow::Schema> sc;
  std::shared_ptr<arrow::Schema>* out = &sc;
  ARROWSTATUS_OK(reader->GetSchema(out));

  int idx = sc -> GetFieldIndex(colname);
  // Since this doesn't actually throw a Parquet error, we have to generate
  // our own error message for this case
  if(idx == -1) {
    std::string fname(filename);
    std::string dname(colname);
    std::string msg = "Dataset: " + dname + " does not exist in file: " + filename; 
    *errMsg = strdup(msg.c_str());
    return ARROWERROR;
  }
  auto myType = sc -> field(idx) -> type();

  if(myType == arrow::int64())
    return ARROWINT64;
  else if(myType == arrow::int32())
    return ARROWINT32;
  else // TODO: error type not supported
    return ARROWUNDEFINED;
}

#define COPYTOCHAPEL(arrowtype, chunk)                                      \
  auto int_arr = std::static_pointer_cast<arrow::arrowtype>(chunk); \
  for(int i = 0; i < numElems; i++) \
    chpl_ptr[i] = int_arr->Value(i);

int cpp_readColumnByName(const char* filename, void* chpl_arr, const char* colname, int numElems, char** errMsg) {
  auto chpl_ptr = (int64_t*)chpl_arr;
  int ty = cpp_getType(filename, colname, errMsg);
  
  // Since Arrow type matches Chapel type, we can batch read
  if(ty == ARROWINT64) {
    std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(filename, false);

    std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
    int num_row_groups = file_metadata->num_row_groups();
    int num_columns = file_metadata->num_columns();

    int i = 0;
    for (int r = 0; r < num_row_groups; ++r) {
      std::shared_ptr<parquet::RowGroupReader> row_group_reader =
        parquet_reader->RowGroup(r);

      int64_t values_read = 0;
      int64_t rows_read = 0;
      int numToRead = 1000;

      std::shared_ptr<parquet::ColumnReader> column_reader;
      ARROW_UNUSED(rows_read);  // prevent warning in release build

      int idx = file_metadata -> schema() -> ColumnIndex(colname);

      if(idx < 0) {
        std::string fname(filename);
        std::string dname(colname);
        std::string msg = "Dataset: " + dname + " does not exist in file: " + filename; 
        *errMsg = strdup(msg.c_str());
        return ARROWERROR;
      }
      
      // Get the Column Reader for the specified column
      column_reader = row_group_reader->Column(idx);
      parquet::Int64Reader* int64_reader =
        static_cast<parquet::Int64Reader*>(column_reader.get());

      // Read all the rows in the column
      while (int64_reader->HasNext()) {
        rows_read = int64_reader->ReadBatch(numToRead, nullptr, nullptr, &chpl_ptr[i], &values_read);
        i+=numToRead;
      }
    }
  } else {
    std::shared_ptr<arrow::io::ReadableFile> infile;
    ARROWRESULT_OK(arrow::io::ReadableFile::Open(filename,arrow::default_memory_pool()),
                   infile);

    std::unique_ptr<parquet::arrow::FileReader> reader;
    ARROWSTATUS_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

    std::shared_ptr<arrow::ChunkedArray> array;

    std::shared_ptr<arrow::Schema> sc;
    std::shared_ptr<arrow::Schema>* out = &sc;
    ARROWSTATUS_OK(reader->GetSchema(out));

    if(!reader->ReadColumn(sc -> GetFieldIndex(colname), &array).ok()) {
      std::string fname(filename);
      std::string dname(colname);
      std::string msg = "Dataset: " + dname + " does not exist in file: " + filename; 
      *errMsg = strdup(msg.c_str());
      return ARROWERROR;
    }

    std::shared_ptr<arrow::Array> regular = array->chunk(0);

    if(ty == ARROWINT32) {
      COPYTOCHAPEL(Int32Array, regular);
    }
  }
  return 0;
}

int cpp_writeColumnToParquet(const char* filename, void* chpl_arr,
                             int colnum, const char* dsetname, int numelems,
                             int rowGroupSize, char** errMsg) {
  auto chpl_ptr = (int64_t*)chpl_arr;
  arrow::Int64Builder i64builder;
  arrow::Status status;
  for(int i = 0; i < numelems; i++) {
    ARROWSTATUS_OK(i64builder.AppendValues({chpl_ptr[i]}));
  }
  std::shared_ptr<arrow::Array> i64array;
  ARROWSTATUS_OK(i64builder.Finish(&i64array));

  std::shared_ptr<arrow::Schema> schema = arrow::schema(
                 {arrow::field(dsetname, arrow::int64())});

  auto table = arrow::Table::Make(schema, {i64array});

  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  ARROWRESULT_OK(arrow::io::FileOutputStream::Open(filename),
                 outfile);

  ARROWSTATUS_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, rowGroupSize));
  return 0;
}

const char* cpp_getVersionInfo(void) {
  return strdup(arrow::GetBuildInfo().version_string.c_str());
}

void cpp_free_string(void* ptr) {
  free(ptr);
}

/*
 C functions
 -----------
 These C functions provide no functionality, since the C++
 Arrow library is being used, they merely call the C++ functions
 to allow Chapel to call the C++ functions through C interoperability.
 Each Arrow function must have a corresponding C function if wished
 to be called by Chapel.
*/

extern "C" {
  int c_getNumRows(const char* chpl_str, char** errMsg) {
    return cpp_getNumRows(chpl_str, errMsg);
  }

  int c_readColumnByName(const char* filename, void* chpl_arr, const char* colname, int numElems, char** errMsg) {
    return cpp_readColumnByName(filename, chpl_arr, colname, numElems, errMsg);
  }

  int c_getType(const char* filename, const char* colname, char** errMsg) {
    return cpp_getType(filename, colname, errMsg);
  }

  int c_writeColumnToParquet(const char* filename, void* chpl_arr,
                             int colnum, const char* dsetname, int numelems,
                             int rowGroupSize, char** errMsg) {
    return cpp_writeColumnToParquet(filename, chpl_arr, colnum, dsetname,
                                    numelems, rowGroupSize, errMsg);
  }

  const char* c_getVersionInfo(void) {
    return cpp_getVersionInfo();
  }

  void c_free_string(void* ptr) {
    cpp_free_string(ptr);
  }
}
