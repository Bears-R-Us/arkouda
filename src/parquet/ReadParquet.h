#ifndef READ_PARQUET_H
#define READ_PARQUET_H

#include <stdint.h>
#include <stdbool.h>

#include "UtilParquet.h"


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

namespace akcpp {
  // Engin: We could consider exposing this struct to Chapel. Note that in
  // ParquetMsg.chpl there is a somewhat similar type. However, this one is for a
  // single column read
  // Based on what we decide, the implementation of this struct can be mode more
  // C++-like or could stay more C-like.
  struct ColReadOp {
    void* chpl_arr;
    int64_t *startIdx;
    std::shared_ptr<parquet::ColumnReader> column_reader;
    bool hasNonFloatNulls;
    chplEnum_t nullMode;
    int64_t row_idx;
    int64_t numElems;
    int64_t batchSize;
    bool* where_null_chpl;
    const parquet::ColumnDescriptor* col_info;

    template<typename ArrowType>
    int64_t read();

    template<typename Types>
    int64_t _readShortIntegral();
  };

  int readAllCols(const char* filename, void** chpl_arrs, int* types,
                  bool* where_null_chpl, int64_t numElems, int64_t startIdx,
                  int64_t batchSize, chplEnum_t nullMode, char** errMsg);
}


extern "C" {
#endif
  int c_readStrColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t batchSize, char** errMsg);
  
  int cpp_readStrColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t batchSize, char** errMsg);

  int c_readAllCols(const char* filename, void** chpl_arrs, int* types,
                         bool* where_null_chpl, int64_t numElems,
                         int64_t startIdx, int64_t batchSize,
                         chplEnum_t nullMode, char** errMsg);

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
#endif //READ_PARQUET_H
