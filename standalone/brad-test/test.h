#include <stdint.h>
#include <stdbool.h>

// Wrap functions in C extern if compiling C++ object file
#ifdef __cplusplus
#include <iostream>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/column_reader.h>
#include <parquet/api/writer.h>
#include <parquet/schema.h>
#include <parquet/types.h>
#include <cmath>
#include <queue>
extern "C" {
#endif

  typedef struct {
    uint32_t len;
    const uint8_t* ptr;
  } MyByteArray;

  void c_openFile(const char* filename, int64_t idx);
  void cpp_openFile(const char* filename, int64_t idx);

  void c_createRowGroupReader(int64_t rowGroup, int64_t readerIdx);
  void cpp_createRowGroupReader(int64_t rowGroup, int64_t readerIdx);

  void c_createColumnReader(const char* colname, int64_t readerIdx);
  void cpp_createColumnReader(const char* colname, int64_t readerIdx);

  void* c_readParquetColumnChunks(const char* filename, int64_t batchSize, int64_t numElems, int64_t readerIdx, int64_t* numRead);
  void* cpp_readParquetColumnChunks(const char* filename, int64_t batchSize, int64_t numElems, int64_t readerIdx, int64_t* numRead);

  int c_getNumRowGroups(int64_t readerIdx);
  int cpp_getNumRowGroups(int64_t readerIdx);

#ifdef __cplusplus
  std::map<int, std::shared_ptr<parquet::ParquetFileReader>> globalFiles;
  std::map<int, std::shared_ptr<parquet::RowGroupReader>> globalRowGroupReaders;
  std::map<int, std::shared_ptr<parquet::ColumnReader>> globalColumnReaders;
}
#endif
