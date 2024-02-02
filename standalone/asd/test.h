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

  void c_getParquetReader(void* reader, const char* filename, const char* colname);
  void cpp_getParquetReader(void* reader, const char* filename, const char* colname);

  void* c_readParquetColumn(void* reader, void* chpl_arr, int64_t batchSize, int64_t numElems);
  void* cpp_readParquetColumn(void* reader, void* chpl_arr, int64_t batchSize, int64_t numElems);  

  void* c_readParquetColumnChunks(const char* filename, const char* colname, int64_t batchSize, int64_t numElems);
  void* cpp_readParquetColumnChunks(const char* filename, const char* colname, int64_t batchSize, int64_t numElems);

#ifdef __cplusplus
}
#endif
