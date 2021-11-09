#include "ArrowFunctions.h"

#include <iostream>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

/*
 C++ functions
 -------------
 These C++ functions are used to call into the Arrow library
 and are then called to by their corresponding C functions to
 allow interoperability with Chapel. This means that all of the
 C++ functions must return types that are C compatible.
*/

int cpp_getNumRows(const char* filename) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(
      infile,
      arrow::io::ReadableFile::Open(filename,
                                    arrow::default_memory_pool()));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(
    parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
  return reader -> parquet_reader() -> metadata() -> num_rows();
}

int cpp_getType(const char* filename, const char* colname) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(
      infile,
      arrow::io::ReadableFile::Open(filename,
                                    arrow::default_memory_pool()));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

  std::shared_ptr<arrow::Schema> sc;
  std::shared_ptr<arrow::Schema>* out = &sc;
  PARQUET_THROW_NOT_OK(reader->GetSchema(out));

  int idx = sc -> GetFieldIndex(colname);
  if(idx == -1) // TODO: error colname not in schema
    idx = 0;
  auto myType = sc -> field(idx) -> type();

  if(myType == arrow::int64())
    return ARROWINT64;
  else if(myType == arrow::int32())
    return ARROWINT32;
  else // TODO: error type not supported
    return ARROWUNDEFINED;
}

void cpp_readColumnByName(const char* filename, void* chpl_arr, const char* colname, int numElems) {
  auto chpl_ptr = (int64_t*)chpl_arr;

  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(
      infile,
      arrow::io::ReadableFile::Open(filename,
                                    arrow::default_memory_pool()));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
  std::shared_ptr<arrow::ChunkedArray> array;

  std::shared_ptr<arrow::Schema> sc;
  std::shared_ptr<arrow::Schema>* out = &sc;
  PARQUET_THROW_NOT_OK(reader->GetSchema(out));

  auto idx = sc -> GetFieldIndex(colname);

  // TODO: error: schema does not contain dsetname
  if(idx == -1)
    idx = 0;

  PARQUET_THROW_NOT_OK(reader->ReadColumn(idx, &array));

  int ty = cpp_getType(filename, colname);
  std::shared_ptr<arrow::Array> regular = array->chunk(0);

  if(ty == ARROWINT64) {
    auto int_arr = std::static_pointer_cast<arrow::Int64Array>(regular);

    for(int i = 0; i < numElems; i++)
      chpl_ptr[i] = int_arr->Value(i);
  } else if(ty == ARROWINT32) {
      auto int_arr = std::static_pointer_cast<arrow::Int32Array>(regular);

      for(int i = 0; i < numElems; i++)
        chpl_ptr[i] = int_arr->Value(i);
  }
}

void cpp_writeColumnToParquet(const char* filename, void* chpl_arr,
                              int colnum, const char* dsetname, int numelems,
                              int rowGroupSize) {
  auto chpl_ptr = (int64_t*)chpl_arr;
  arrow::Int64Builder i64builder;
  for(int i = 0; i < numelems; i++)
    PARQUET_THROW_NOT_OK(i64builder.AppendValues({chpl_ptr[i]}));
  std::shared_ptr<arrow::Array> i64array;
  PARQUET_THROW_NOT_OK(i64builder.Finish(&i64array));

  std::shared_ptr<arrow::Schema> schema = arrow::schema(
                 {arrow::field(dsetname, arrow::int64())});

  auto table = arrow::Table::Make(schema, {i64array});

  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  PARQUET_ASSIGN_OR_THROW(
      outfile,
      arrow::io::FileOutputStream::Open(filename));

  PARQUET_THROW_NOT_OK(
      parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, rowGroupSize));
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
  int c_getNumRows(const char* chpl_str) {
    return cpp_getNumRows(chpl_str);
  }

  void c_readColumnByName(const char* filename, void* chpl_arr, const char* colname, int numElems) {
    cpp_readColumnByName(filename, chpl_arr, colname, numElems);
  }

  int c_getType(const char* filename, const char* colname) {
    return cpp_getType(filename, colname);
  }

  void c_writeColumnToParquet(const char* filename, void* chpl_arr,
                              int colnum, const char* dsetname, int numelems,
                              int rowGroupSize) {
    cpp_writeColumnToParquet(filename, chpl_arr, colnum, dsetname,
                             numelems, rowGroupSize);
  }

  const char* c_getVersionInfo(void) {
    return cpp_getVersionInfo();
  }

  void c_free_string(void* ptr) {
    cpp_free_string(ptr);
  }
}
