#include "ReadParquet.h"
#include "UtilParquet.h"

// Returns the number of elements read
template <typename ReaderType, typename ChplType>
int64_t readColumn(void* chpl_arr, int64_t *startIdx, std::shared_ptr<parquet::ColumnReader> column_reader,
                bool hasNonFloatNulls, int64_t i, int64_t numElems, int64_t batchSize,
                int64_t values_read, bool* where_null_chpl) {
  int16_t definition_level; // nullable type and only reading single records in batch
  auto chpl_ptr = (ChplType*)chpl_arr;
  int64_t num_read = 0;
  ReaderType* reader =
    static_cast<ReaderType*>(column_reader.get());
  *startIdx -= reader->Skip(*startIdx);

  if (not hasNonFloatNulls) {
    while (reader->HasNext() && i < numElems) {
      if((numElems - i) < batchSize) // adjust batchSize if needed
        batchSize = numElems - i;
      (void)reader->ReadBatch(batchSize, nullptr, nullptr, &chpl_ptr[i], &values_read);
      i+=values_read;
      num_read += values_read;
    }
  }
  else {
    while (reader->HasNext() && i < numElems) {
      (void)reader->ReadBatch(1, &definition_level, nullptr, &chpl_ptr[i], &values_read);
      // if values_read is 0, that means that it was a null value
      if(values_read == 0) {
        where_null_chpl[i] = true;
      }
      i++;
      num_read++;
    }
  }
  return num_read;
}

template <typename ReaderType, typename ChplType, typename PqType>
int64_t readColumnDbFl(void* chpl_arr, int64_t *startIdx, std::shared_ptr<parquet::ColumnReader> column_reader,
                    bool hasNonFloatNulls, int64_t i, int64_t numElems, int64_t batchSize,
                    int64_t values_read, bool* where_null_chpl) {
  int16_t definition_level; // nullable type and only reading single records in batch
  auto chpl_ptr = (ChplType*)chpl_arr;
  ReaderType* reader =
    static_cast<ReaderType*>(column_reader.get());
  *startIdx -= reader->Skip(*startIdx);

  int64_t num_read = 0;
  while (reader->HasNext() && i < numElems) {
    PqType value;
    (void)reader->ReadBatch(1, &definition_level, nullptr, &value, &values_read);
    // if values_read is 0, that means that it was a null value
    if(values_read > 0) {
      // this means it wasn't null
      chpl_ptr[i] = (ChplType) value;
    } else {
      chpl_ptr[i] = NAN;
    }
    i++;
    num_read++;
  }
  return num_read;
}

template <typename ReaderType, typename ChplType, typename PqType>
int64_t readColumnIrregularBitWidth(void* chpl_arr, int64_t *startIdx, std::shared_ptr<parquet::ColumnReader> column_reader,
                                 bool hasNonFloatNulls, int64_t i, int64_t numElems, int64_t batchSize,
                                 int64_t values_read, bool* where_null_chpl) {
  int16_t definition_level; // nullable type and only reading single records in batch
  auto chpl_ptr = (ChplType*)chpl_arr;
  ReaderType* reader =
    static_cast<ReaderType*>(column_reader.get());
  *startIdx -= reader->Skip(*startIdx);

  int64_t num_read = 0;
  if (not hasNonFloatNulls) {
    PqType* tmpArr = (PqType*)malloc(batchSize * sizeof(int32_t));
    while (reader->HasNext() && i < numElems) {
      if((numElems - i) < batchSize) // adjust batchSize if needed
        batchSize = numElems - i;

      // Can't read directly into chpl_ptr because it is int64
      (void)reader->ReadBatch(batchSize, nullptr, nullptr, tmpArr, &values_read);
      for (int64_t j = 0; j < values_read; j++)
        chpl_ptr[i+j] = (ChplType)tmpArr[j];
      i+=values_read;
      num_read+=values_read;
    }
    free(tmpArr);
  }
  else {
    int32_t tmp;
    while (reader->HasNext() && i < numElems) {
      (void)reader->ReadBatch(1, &definition_level, nullptr, &tmp, &values_read);
      // if values_read is 0, that means that it was a null value
      if(values_read == 0) {
        where_null_chpl[i] = true;
      }
      else {
        chpl_ptr[i] = (int64_t)tmp;
      }
      i++;
      num_read++;
    }
  }
  return num_read;
}

int cpp_readStrColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t batchSize, char** errMsg) {
  try {
    int64_t ty = cpp_getType(filename, colname, errMsg);
  
    std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(filename, false);

    std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
    int num_row_groups = file_metadata->num_row_groups();

    int64_t i = 0;
    for (int r = 0; r < num_row_groups; r++) {
      std::shared_ptr<parquet::RowGroupReader> row_group_reader =
        parquet_reader->RowGroup(r);

      int64_t values_read = 0;

      std::shared_ptr<parquet::ColumnReader> column_reader;

      auto idx = file_metadata -> schema() -> ColumnIndex(colname);
      auto max_def = file_metadata -> schema() -> Column(idx) -> max_definition_level(); // needed to determine if nulls are allowed

      if(idx < 0) {
        std::string dname(colname);
        std::string fname(filename);
        std::string msg = "Dataset: " + dname + " does not exist in file: " + fname; 
        *errMsg = strdup(msg.c_str());
        return ARROWERROR;
      }
      
      column_reader = row_group_reader->Column(idx);

      if(ty == ARROWSTRING) {
        auto chpl_ptr = (unsigned char*)chpl_arr;
        parquet::ByteArrayReader* reader =
          static_cast<parquet::ByteArrayReader*>(column_reader.get());

        int totalProcessed = 0;
        std::vector<parquet::ByteArray> values(batchSize);
        while (reader->HasNext() && totalProcessed < numElems) {
          std::vector<int16_t> definition_levels(batchSize,-1);
          if((numElems - totalProcessed) < batchSize) // adjust batchSize if needed
            batchSize = numElems - totalProcessed;
          
          (void)reader->ReadBatch(batchSize, definition_levels.data(), nullptr, values.data(), &values_read);
          totalProcessed += values_read;
          int j = 0;
          int numProcessed = 0;
          while(j < batchSize) {
            if(definition_levels[j] == 1) {
              for(int k = 0; k < values[numProcessed].len; k++) {
                chpl_ptr[i] = values[numProcessed].ptr[k];
                i++;
              }
              i++; // skip one space so the strings are null terminated with a 0
              numProcessed++;
            } else if(definition_levels[j] == 0) {
              i++;
            } else {
              j = batchSize; // exit loop, not read
            }
            j++;
          }
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int cpp_readColumnByName(const char* filename, void* chpl_arr, bool* where_null_chpl, const char* colname, int64_t numElems, int64_t startIdx, int64_t batchSize, int64_t byteLength, bool hasNonFloatNulls, char** errMsg) {
  try {
    int64_t ty = cpp_getType(filename, colname, errMsg);
  
    std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(filename, false);

    std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
    int num_row_groups = file_metadata->num_row_groups();

    int64_t i = 0;
    for (int r = 0; (r < num_row_groups) && (i < numElems); r++) {
      std::shared_ptr<parquet::RowGroupReader> row_group_reader =
        parquet_reader->RowGroup(r);

      int64_t values_read = 0;

      std::shared_ptr<parquet::ColumnReader> column_reader;

      auto idx = file_metadata -> schema() -> ColumnIndex(colname);

      if(idx < 0) {
        std::string dname(colname);
        std::string fname(filename);
        std::string msg = "Dataset: " + dname + " does not exist in file: " + fname; 
        *errMsg = strdup(msg.c_str());
        return ARROWERROR;
      }
      auto max_def = file_metadata -> schema() -> Column(idx) -> max_definition_level(); // needed to determine if nulls are allowed
      
      column_reader = row_group_reader->Column(idx);

      // Since int64 and uint64 Arrow dtypes share a physical type and only differ
      // in logical type, they must be read from the file in the same way
      if(ty == ARROWINT64 || ty == ARROWUINT64) {
        i += readColumn<parquet::Int64Reader, int64_t>(chpl_arr, &startIdx, column_reader, hasNonFloatNulls, i,
                   numElems, batchSize, values_read, where_null_chpl);
      } else if(ty == ARROWINT32 || ty == ARROWUINT32) {
        i += readColumnIrregularBitWidth<parquet::Int32Reader, int64_t, int32_t>(chpl_arr, &startIdx, column_reader, hasNonFloatNulls, i,
                                              numElems, batchSize, values_read, where_null_chpl);
      } else if(ty == ARROWBOOLEAN) {
        i += readColumn<parquet::BoolReader, bool>(chpl_arr, &startIdx, column_reader, hasNonFloatNulls, i,
                                              numElems, batchSize, values_read, where_null_chpl);
      } else if(ty == ARROWSTRING) {
        int16_t definition_level; // nullable type and only reading single records in batch
        auto chpl_ptr = (unsigned char*)chpl_arr;
        parquet::ByteArrayReader* reader =
          static_cast<parquet::ByteArrayReader*>(column_reader.get());

        while (reader->HasNext()) {
          parquet::ByteArray value;
          (void)reader->ReadBatch(1, &definition_level, nullptr, &value, &values_read);
          // if values_read is 0, that means that it was a null value
          if(values_read > 0) {
            for(int j = 0; j < value.len; j++) {
              chpl_ptr[i] = value.ptr[j];
              i++;
            }
          }
          i++; // skip one space so the strings are null terminated with a 0
        }        
      } else if(ty == ARROWFLOAT) {
        i += readColumnDbFl<parquet::FloatReader, double, float>(chpl_arr, &startIdx, column_reader, hasNonFloatNulls, i,
                                                            numElems, batchSize, values_read, where_null_chpl);
      } else if(ty == ARROWDOUBLE) {
        i += readColumnDbFl<parquet::DoubleReader, double, double>(chpl_arr, &startIdx, column_reader, hasNonFloatNulls, i,
                                                            numElems, batchSize, values_read, where_null_chpl);
      } else if(ty == ARROWDECIMAL) {
        auto chpl_ptr = (double*)chpl_arr;
        parquet::FixedLenByteArray value;
        parquet::FixedLenByteArrayReader* reader =
          static_cast<parquet::FixedLenByteArrayReader*>(column_reader.get());
        startIdx -= reader->Skip(startIdx);

        while (reader->HasNext() && i < numElems) {
          (void)reader->ReadBatch(1, nullptr, nullptr, &value, &values_read);
          arrow::Decimal128 v;
          PARQUET_ASSIGN_OR_THROW(v,
                                  ::arrow::Decimal128::FromBigEndian(value.ptr, byteLength));

          chpl_ptr[i] = v.ToDouble(0);
          i+=values_read;
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int cpp_readListColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
  try {
    int64_t ty = cpp_getType(filename, colname, errMsg);
    if (ty == ARROWLIST){
      int64_t lty = cpp_getListType(filename, colname, errMsg);
      std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
        parquet::ParquetFileReader::OpenFile(filename, false);

      std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
      int num_row_groups = file_metadata->num_row_groups();

      auto idx = file_metadata -> schema() -> group_node() -> FieldIndex(colname);
      if(idx < 0) {
        std::string dname(colname);
        std::string fname(filename);
        std::string msg = "Dataset: " + dname + " does not exist in file: " + fname; 
        *errMsg = strdup(msg.c_str());
        return ARROWERROR;
      }

      int64_t i = 0;
      int64_t arrayIdx = 0;
      for (int r = 0; r < num_row_groups; r++) {
        std::shared_ptr<parquet::RowGroupReader> row_group_reader =
          parquet_reader->RowGroup(r);

        int64_t values_read = 0;
        int16_t definition_level; // needed for any type that is nullable

        std::shared_ptr<parquet::ColumnReader> column_reader = row_group_reader->Column(idx);
        if(lty == ARROWINT64 || lty == ARROWUINT64) {
          int16_t definition_level; // nullable type and only reading single records in batch
          auto chpl_ptr = (int64_t*)chpl_arr;
          parquet::Int64Reader* reader =
            static_cast<parquet::Int64Reader*>(column_reader.get());
          startIdx -= reader->Skip(startIdx);

          while (reader->HasNext() && arrayIdx < numElems) {
            (void)reader->ReadBatch(1, &definition_level, nullptr, &chpl_ptr[arrayIdx], &values_read);
            // if values_read is 0, that means that it was an empty seg
            if (values_read != 0) {
              arrayIdx++;
            }
            i++;
          }
        } else if(lty == ARROWINT32 || lty == ARROWUINT32) {
          int16_t definition_level; // nullable type and only reading single records in batch
          auto chpl_ptr = (int64_t*)chpl_arr;
          parquet::Int32Reader* reader =
            static_cast<parquet::Int32Reader*>(column_reader.get());
          startIdx -= reader->Skip(startIdx);

          int32_t tmp;
          while (reader->HasNext() && arrayIdx < numElems) {
            (void)reader->ReadBatch(1, &definition_level, nullptr, &tmp, &values_read);
            // if values_read is 0, that means that it was an empty seg
            if (values_read != 0) {
              chpl_ptr[arrayIdx] = (int64_t)tmp;
              arrayIdx++;
            }
            i++;
          }
        } else if (lty == ARROWSTRING) {
          int16_t definition_level; // nullable type and only reading single records in batch
          auto chpl_ptr = (unsigned char*)chpl_arr;
          parquet::ByteArrayReader* reader =
            static_cast<parquet::ByteArrayReader*>(column_reader.get());

          while (reader->HasNext()) {
            parquet::ByteArray value;
            (void)reader->ReadBatch(1, &definition_level, nullptr, &value, &values_read);
            // if values_read is 0, that means that it was a null value
            if(values_read > 0 && definition_level == 3) {
              for(int j = 0; j < value.len; j++) {
                chpl_ptr[i] = value.ptr[j];
                i++;
              }
              i++; // skip one space so the strings are null terminated with a 0
            }
          }
        } else if(lty == ARROWBOOLEAN) {
          int16_t definition_level; // nullable type and only reading single records in batch
          auto chpl_ptr = (bool*)chpl_arr;
          parquet::BoolReader* reader =
            static_cast<parquet::BoolReader*>(column_reader.get());
          startIdx -= reader->Skip(startIdx);

          while (reader->HasNext() && arrayIdx < numElems) {
            (void)reader->ReadBatch(1, &definition_level, nullptr, &chpl_ptr[arrayIdx], &values_read);
            // if values_read is 0, that means that it was an empty seg
            if (values_read != 0) {
              arrayIdx++;
            }
            i++;
          }
        } else if(lty == ARROWFLOAT) {
          // convert to simpler single batch to sidestep this seemingly architecture dependent (see issue #3234)
          int16_t definition_level; // nullable type and only reading single records in batch
          auto chpl_ptr = (double*)chpl_arr;
          parquet::FloatReader* reader =
            static_cast<parquet::FloatReader*>(column_reader.get());

          float tmp;
          while (reader->HasNext() && arrayIdx < numElems) {
            (void)reader->ReadBatch(1, &definition_level, nullptr, &tmp, &values_read);
            // if values_read is 0, that means that it was a null value or empty seg
            if (values_read != 0) {
              chpl_ptr[arrayIdx] = (double) tmp;
              arrayIdx++;
            }
            else {
              // check if nan otherwise it's an empty seg
              if (definition_level == 2) {
                chpl_ptr[arrayIdx] = NAN;
                arrayIdx++;
              }
            }
            i++;
          }
        } else if(lty == ARROWDOUBLE) {
          // convert to simpler single batch to sidestep this seemingly architecture dependent (see issue #3234)
          int16_t definition_level; // nullable type and only reading single records in batch
          auto chpl_ptr = (double*)chpl_arr;
          parquet::DoubleReader* reader =
            static_cast<parquet::DoubleReader*>(column_reader.get());

          while (reader->HasNext() && arrayIdx < numElems) {
            (void)reader->ReadBatch(1, &definition_level, nullptr, &chpl_ptr[arrayIdx], &values_read);
            // if values_read is 0, that means that it was a null value or empty seg
            if (values_read != 0) {
              arrayIdx++;
            }
            else {
              // check if nan otherwise it's an empty seg
              if (definition_level == 2) {
                chpl_ptr[arrayIdx] = NAN;
                arrayIdx++;
              }
            }
            i++;
          }
        }
      }
      return 0;
    }
    return ARROWERROR;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int64_t cpp_getStringColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
  try {
    int64_t ty = cpp_getType(filename, colname, errMsg);
    auto offsets = (int64_t*)chpl_offsets;
    int64_t byteSize = 0;

    if(ty == ARROWSTRING) {
      std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
        parquet::ParquetFileReader::OpenFile(filename, false);

      std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
      int num_row_groups = file_metadata->num_row_groups();

      int64_t i = 0;
      for (int r = 0; r < num_row_groups; r++) {
        std::shared_ptr<parquet::RowGroupReader> row_group_reader =
          parquet_reader->RowGroup(r);

        int64_t values_read = 0;

        std::shared_ptr<parquet::ColumnReader> column_reader;

        int64_t idx;
        idx = file_metadata -> schema() -> ColumnIndex(colname);

        if(idx < 0) {
          std::string dname(colname);
          std::string fname(filename);
          std::string msg = "Dataset: " + dname + " does not exist in file: " + fname; 
          *errMsg = strdup(msg.c_str());
          return ARROWERROR;
        }
        column_reader = row_group_reader->Column(idx);

        int16_t definition_level;
        parquet::ByteArrayReader* ba_reader =
          static_cast<parquet::ByteArrayReader*>(column_reader.get());

        int64_t numRead = 0;
        
        int totalProcessed = 0;
        std::vector<parquet::ByteArray> values(batchSize);
        while (ba_reader->HasNext() && totalProcessed < numElems) {
          if((numElems - totalProcessed) < batchSize) // adjust batchSize if needed
            batchSize = numElems - totalProcessed;
          std::vector<int16_t> definition_levels(batchSize,-1);
          (void)ba_reader->ReadBatch(batchSize, definition_levels.data(), nullptr, values.data(), &values_read);
          totalProcessed += values_read;
          int j = 0;
          int numProcessed = 0;
          while(j < batchSize) {
            if(definition_levels[j] == 1 || definition_levels[j] == 3) {
              offsets[i] = values[numProcessed].len + 1;
              byteSize += values[numProcessed].len + 1;
              numProcessed++;
              i++;
            } else if(definition_levels[j] == 0) {
              offsets[i] = 1;
              byteSize+=1;
              i++;
            } else {
              j = batchSize; // exit condition
            }
            j++;
          }
        }
      }
      return byteSize;
    }
    return ARROWERROR;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int64_t cpp_getStringListColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
  try {
    int64_t ty = cpp_getType(filename, colname, errMsg);
    int64_t dty; // used to store the type of data so we can handle lists
    if (ty == ARROWLIST) { // get the type of the list so we can verify it is ARROWSTRING
      dty = cpp_getListType(filename, colname, errMsg);
    }
    else {
      dty = ty;
    }
    auto offsets = (int64_t*)chpl_offsets;
    int64_t byteSize = 0;

    if(dty == ARROWSTRING) {
      std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
        parquet::ParquetFileReader::OpenFile(filename, false);

      std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
      int num_row_groups = file_metadata->num_row_groups();

      int64_t i = 0;
      for (int r = 0; r < num_row_groups; r++) {
        std::shared_ptr<parquet::RowGroupReader> row_group_reader =
          parquet_reader->RowGroup(r);

        int64_t values_read = 0;

        std::shared_ptr<parquet::ColumnReader> column_reader;

        int64_t idx;
        if (ty == ARROWLIST) {
          idx = file_metadata -> schema() -> group_node() -> FieldIndex(colname);
        } else {
          idx = file_metadata -> schema() -> ColumnIndex(colname);
        }

        if(idx < 0) {
          std::string dname(colname);
          std::string fname(filename);
          std::string msg = "Dataset: " + dname + " does not exist in file: " + fname; 
          *errMsg = strdup(msg.c_str());
          return ARROWERROR;
        }
        column_reader = row_group_reader->Column(idx);

        int16_t definition_level;
        parquet::ByteArrayReader* ba_reader =
          static_cast<parquet::ByteArrayReader*>(column_reader.get());

        int64_t numRead = 0;
        while (ba_reader->HasNext() && numRead < numElems) {
          parquet::ByteArray value;
          (void)ba_reader->ReadBatch(1, &definition_level, nullptr, &value, &values_read);
          if ((ty == ARROWLIST && definition_level == 3) || ty == ARROWSTRING) {
            if(values_read > 0) {
              offsets[i] = value.len + 1;
              byteSize += value.len + 1;
              numRead += values_read;
            } else {
              offsets[i] = 1;
              byteSize+=1;
              numRead+=1;
            }
            i++;
          }
        }
      }
      return byteSize;
    }
    return ARROWERROR;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int64_t cpp_getListColumnSize(const char* filename, const char* colname, void* chpl_seg_sizes, int64_t numElems, int64_t startIdx, char** errMsg) {
  try {
    int64_t ty = cpp_getType(filename, colname, errMsg);
    auto seg_sizes = (int64_t*)chpl_seg_sizes;
    int64_t listSize = 0;
    
    if (ty == ARROWLIST){
      int64_t lty = cpp_getListType(filename, colname, errMsg);
      std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
        parquet::ParquetFileReader::OpenFile(filename, false);

      std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
      int num_row_groups = file_metadata->num_row_groups();

      auto idx = file_metadata -> schema() -> group_node() -> FieldIndex(colname);
      if(idx < 0) {
        std::string dname(colname);
        std::string fname(filename);
        std::string msg = "Dataset: " + dname + " does not exist in file: " + fname; 
        *errMsg = strdup(msg.c_str());
        return ARROWERROR;
      }

      int64_t i = 0;
      int64_t vct = 0;
      int64_t seg_size = 0;
      int64_t off = 0;
      bool first = true;
      for (int r = 0; r < num_row_groups; r++) {
        std::shared_ptr<parquet::RowGroupReader> row_group_reader =
          parquet_reader->RowGroup(r);

        int64_t values_read = 0;

        std::shared_ptr<parquet::ColumnReader> column_reader;

        column_reader = row_group_reader->Column(idx);
        int16_t definition_level;
        int16_t rep_lvl;

        if(lty == ARROWINT64 || lty == ARROWUINT64) {
          parquet::Int64Reader* int_reader =
            static_cast<parquet::Int64Reader*>(column_reader.get());

          while (int_reader->HasNext()) {
            int64_t value;
            (void)int_reader->ReadBatch(1, &definition_level, &rep_lvl, &value, &values_read);
            if (values_read == 0 || (!first && rep_lvl == 0)) {
              seg_sizes[i] = seg_size;
              i++;
              seg_size = 0;
            }
            if (values_read != 0) {
              seg_size++;
              vct++;
              if (first) {
                first = false;
              }
            }
            if (values_read != 0 && !int_reader->HasNext()){
              seg_sizes[i] = seg_size;
            }
          }
        } else if(lty == ARROWINT32 || lty == ARROWUINT32) {
          parquet::Int32Reader* int_reader =
            static_cast<parquet::Int32Reader*>(column_reader.get());

          while (int_reader->HasNext()) {
            int32_t value;
            (void)int_reader->ReadBatch(1, &definition_level, &rep_lvl, &value, &values_read);
            if (values_read == 0 || (!first && rep_lvl == 0)) {
              seg_sizes[i] = seg_size;
              i++;
              seg_size = 0;
            }
            if (values_read != 0) {
              seg_size++;
              vct++;
              if (first) {
                first = false;
              }
            }
            if (values_read != 0 && !int_reader->HasNext()){
              seg_sizes[i] = seg_size;
            }
          }
        } else if (lty == ARROWSTRING) {
          parquet::ByteArrayReader* reader =
            static_cast<parquet::ByteArrayReader*>(column_reader.get());

          while (reader->HasNext()) {
            parquet::ByteArray value;
            (void)reader->ReadBatch(1, &definition_level, &rep_lvl, &value, &values_read);
            if (values_read == 0 || (!first && rep_lvl == 0)) {
              seg_sizes[i] = seg_size;
              i++;
              seg_size = 0;
            }
            if (values_read != 0) {
              seg_size++;
              vct++;
              if (first) {
                first = false;
              }
            }
            if (values_read != 0 && !reader->HasNext()){
              seg_sizes[i] = seg_size;
            }
          }
        } else if(lty == ARROWBOOLEAN) {
          parquet::BoolReader* bool_reader =
            static_cast<parquet::BoolReader*>(column_reader.get());

          while (bool_reader->HasNext()) {
            bool value;
            (void)bool_reader->ReadBatch(1, &definition_level, &rep_lvl, &value, &values_read);
            if (values_read == 0 || (!first && rep_lvl == 0)) {
              seg_sizes[i] = seg_size;
              i++;
              seg_size = 0;
            }
            if (values_read != 0) {
              seg_size++;
              vct++;
              if (first) {
                first = false;
              }
            }
            if (values_read != 0 && !bool_reader->HasNext()){
              seg_sizes[i] = seg_size;
            }
          }
        } else if (lty == ARROWFLOAT) {
          parquet::FloatReader* float_reader =
            static_cast<parquet::FloatReader*>(column_reader.get());

          int64_t numRead = 0;
          while (float_reader->HasNext()) {
            float value;
            (void)float_reader->ReadBatch(1, &definition_level, &rep_lvl, &value, &values_read);
            if ((values_read == 0 && definition_level != 2) || (!first && rep_lvl == 0)) {
              seg_sizes[i] = seg_size;
              i++;
              seg_size = 0;
            }
            if (values_read != 0 || (values_read == 0 && definition_level == 2)) {
              seg_size++;
              vct++;
              if (first) {
                first = false;
              }
            }
            if ((values_read != 0 || (values_read == 0 && definition_level == 2)) && !float_reader->HasNext()){
              seg_sizes[i] = seg_size;
            }
          }
        } else if(lty == ARROWDOUBLE) {
          parquet::DoubleReader* dbl_reader =
            static_cast<parquet::DoubleReader*>(column_reader.get());

          while (dbl_reader->HasNext()) {
            double value;
            (void)dbl_reader->ReadBatch(1, &definition_level, &rep_lvl, &value, &values_read);
            if ((values_read == 0 && definition_level != 2) || (!first && rep_lvl == 0)) {
              seg_sizes[i] = seg_size;
              i++;
              seg_size = 0;
            }
            if (values_read != 0 || (values_read == 0 && definition_level == 2)) {
              seg_size++;
              vct++;
              if (first) {
                first = false;
              }
            }
            if ((values_read != 0 || (values_read == 0 && definition_level == 2)) && !dbl_reader->HasNext()){
              seg_sizes[i] = seg_size;
            }
          }
        }
      }
      return vct;
    }
    return ARROWERROR;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

extern "C" {
  int c_readStrColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t batchSize, char** errMsg) {
    return cpp_readStrColumnByName(filename, chpl_arr, colname, numElems, batchSize, errMsg);
  }
  
  int c_readColumnByName(const char* filename, void* chpl_arr, bool* where_null_chpl, const char* colname, int64_t numElems, int64_t startIdx, int64_t batchSize, int64_t byteLength, bool hasNonFloatNulls, char** errMsg) {
    return cpp_readColumnByName(filename, chpl_arr, where_null_chpl, colname, numElems, startIdx, batchSize, byteLength, hasNonFloatNulls, errMsg);
  }

  int c_readListColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
    return cpp_readListColumnByName(filename, chpl_arr, colname, numElems, startIdx, batchSize, errMsg);
  }

  int64_t c_getStringColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
    return cpp_getStringColumnNumBytes(filename, colname, chpl_offsets, numElems, startIdx, batchSize, errMsg);
  }

  int64_t c_getListColumnSize(const char* filename, const char* colname, void* chpl_seg_sizes, int64_t numElems, int64_t startIdx, char** errMsg) {
    return cpp_getListColumnSize(filename, colname, chpl_seg_sizes, numElems, startIdx, errMsg);
  }

  int64_t c_getStringListColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
    return cpp_getStringListColumnNumBytes(filename, colname, chpl_offsets, numElems, startIdx, batchSize, errMsg);
  }
}
