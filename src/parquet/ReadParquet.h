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
#include <cmath>
#include <queue>
extern "C" {
#endif
  int c_readStrColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t batchSize, char** errMsg);
  
  int cpp_readStrColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t batchSize, char** errMsg);

  int c_readColumnByName(const char* filename, void* chpl_arr, bool* where_null_chpl,
                         const char* colname, int64_t numElems, int64_t startIdx,
                         int64_t batchSize, int64_t byteLength, bool hasNonFloatNulls, char** errMsg);
  int cpp_readColumnByName(const char* filename, void* chpl_arr, bool* where_null_chpl,
                           const char* colname, int64_t numElems, int64_t startIdx,
                           int64_t batchSize, int64_t byteLength, bool hasNonFloatNulls, char** errMsg);

  int c_readListColumnByName(const char* filename, void* chpl_arr, 
                            const char* colname, int64_t numElems, 
                            int64_t startIdx, int64_t batchSize, char** errMsg);
  int cpp_readListColumnByName(const char* filename, void* chpl_arr, 
                              const char* colname, int64_t numElems, 
                              int64_t startIdx, int64_t batchSize, char** errMsg);

  int64_t cpp_getStringColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets,
                                      int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg);
  int64_t c_getStringColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets,
                                      int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg);

  int64_t c_getListColumnSize(const char* filename, const char* colname,
                                    void* chpl_seg_sizes, int64_t numElems, int64_t startIdx, char** errMsg);
  int64_t cpp_getListColumnSize(const char* filename, const char* colname,
                                    void* chpl_seg_sizes, int64_t numElems, int64_t startIdx, char** errMsg);

  int64_t c_getStringListColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg);
  int64_t cpp_getStringListColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg);
   
#ifdef __cplusplus
}
#endif
