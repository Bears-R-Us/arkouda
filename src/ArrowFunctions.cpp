#include "ArrowFunctions.h"

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

int64_t cpp_getNumRows(const char* filename, char** errMsg) {
  try {
    std::shared_ptr<arrow::io::ReadableFile> infile;
    ARROWRESULT_OK(arrow::io::ReadableFile::Open(filename, arrow::default_memory_pool()),
                   infile);

    std::unique_ptr<parquet::arrow::FileReader> reader;
    ARROWSTATUS_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
  
    return reader -> parquet_reader() -> metadata() -> num_rows();
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int cpp_getType(const char* filename, const char* colname, char** errMsg) {
  try {
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

    if(myType->id() == arrow::Type::INT64)
      return ARROWINT64;
    else if(myType->id() == arrow::Type::INT32 || myType->id() == arrow::Type::INT16)
      return ARROWINT32; // int16 is logical type, stored as int32
    else if(myType->id() == arrow::Type::UINT64)
      return ARROWUINT64;
    else if(myType->id() == arrow::Type::UINT32 || 
            myType->id() == arrow::Type::UINT16)
      return ARROWUINT32; // uint16 is logical type, stored as uint32
    else if(myType->id() == arrow::Type::TIMESTAMP)
      return ARROWTIMESTAMP;
    else if(myType->id() == arrow::Type::BOOL)
      return ARROWBOOLEAN;
    else if(myType->id() == arrow::Type::STRING ||
            myType->id() == arrow::Type::BINARY)
      return ARROWSTRING;
    else if(myType->id() == arrow::Type::FLOAT)
      return ARROWFLOAT;
    else if(myType->id() == arrow::Type::DOUBLE)
      return ARROWDOUBLE;
    else if(myType->id() == arrow::Type::LIST)
      return ARROWLIST;
    else if(myType->id() == arrow::Type::DECIMAL)
      return ARROWDECIMAL;
    else {
      std::string fname(filename);
      std::string dname(colname);
      std::string msg = "Unsupported type on column: " + dname + " in " + filename; 
      *errMsg = strdup(msg.c_str());
      return ARROWERROR;
    }
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int cpp_getListType(const char* filename, const char* colname, char** errMsg) {
  try {
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
      std::string msg = "Dataset: " + dname + " does not exist in file: " + fname; 
      *errMsg = strdup(msg.c_str());
      return ARROWERROR;
    }
    auto myType = sc -> field(idx) -> type();

    if (myType->id() == arrow::Type::LIST) {
      if (myType->num_fields() != 1) {
        std::string fname(filename);
        std::string dname(colname);
        std::string msg = "Column " + dname + " in " + fname + " cannot be read by Arkouda."; 
        *errMsg = strdup(msg.c_str());
        return ARROWERROR;
      }
      else {
        // fields returns a vector of fields, but here we are expecting lists so should only contain 1 item here
        auto field = myType->fields()[0];
        auto f_type = field->type();
        if(f_type->id() == arrow::Type::INT64)
          return ARROWINT64;
        else if(f_type->id() == arrow::Type::INT32 || f_type->id() == arrow::Type::INT16)
          return ARROWINT32;
        else if(f_type->id() == arrow::Type::UINT64)
          return ARROWUINT64;
        else if(f_type->id() == arrow::Type::UINT32 || f_type->id() == arrow::Type::UINT16)
          return ARROWUINT32;
        else if(f_type->id() == arrow::Type::TIMESTAMP)
          return ARROWTIMESTAMP;
        else if(f_type->id() == arrow::Type::BOOL)
          return ARROWBOOLEAN;
        else if(f_type->id() == arrow::Type::STRING ||
                f_type->id() == arrow::Type::BINARY)  // Verify that this is functional as expected
          return ARROWSTRING;
        else if(f_type->id() == arrow::Type::FLOAT)
          return ARROWFLOAT;
        else if(f_type->id() == arrow::Type::DOUBLE)
          return ARROWDOUBLE;
        else {
          std::string fname(filename);
          std::string dname(colname);
          std::string msg = "Unsupported type on column: " + dname + " in " + fname; 
          *errMsg = strdup(msg.c_str());
          return ARROWERROR;
        }
      }
    }
    else {
      std::string fname(filename);
      std::string dname(colname);
      std::string msg = "Column " + dname + " in " + fname + " is not a List"; 
      *errMsg = strdup(msg.c_str());
      return ARROWERROR;
    }
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int64_t cpp_getStringColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, char** errMsg) {
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

int64_t cpp_getStringColumnNullIndices(const char* filename, const char* colname, void* chpl_nulls, char** errMsg) {
  try {
    int64_t ty = cpp_getType(filename, colname, errMsg);
    auto null_indices = (int64_t*)chpl_nulls;
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

        auto idx = file_metadata -> schema() -> ColumnIndex(colname);

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

        while (ba_reader->HasNext()) {
          parquet::ByteArray value;
          (void)ba_reader->ReadBatch(1, &definition_level, nullptr, &value, &values_read);
          if(values_read == 0)
            null_indices[i] = 1;
          i++;
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
      for (int r = 0; r < num_row_groups; r++) {
        std::shared_ptr<parquet::RowGroupReader> row_group_reader =
          parquet_reader->RowGroup(r);

        int64_t values_read = 0;
        int16_t definition_level; // needed for any type that is nullable

        std::shared_ptr<parquet::ColumnReader> column_reader = row_group_reader->Column(idx);
        if(lty == ARROWINT64 || lty == ARROWUINT64) {
          auto chpl_ptr = (int64_t*)chpl_arr;
          parquet::Int64Reader* reader =
            static_cast<parquet::Int64Reader*>(column_reader.get());

          while (reader->HasNext() && i < numElems) {
            if((numElems - i) < batchSize)
              batchSize = numElems - i;
            (void)reader->ReadBatch(batchSize, nullptr, nullptr, &chpl_ptr[i], &values_read);
            i+=values_read;
          }
        } else if(lty == ARROWINT32 || lty == ARROWUINT32) {
          auto chpl_ptr = (int64_t*)chpl_arr;
          parquet::Int32Reader* reader =
            static_cast<parquet::Int32Reader*>(column_reader.get());

          int32_t* tmpArr = (int32_t*)malloc(batchSize * sizeof(int32_t));
          while (reader->HasNext() && i < numElems) {
            if((numElems - i) < batchSize)
              batchSize = numElems - i;
            // Can't read directly into chpl_ptr because it is int64
            (void)reader->ReadBatch(batchSize, nullptr, nullptr, tmpArr, &values_read);
            for (int64_t j = 0; j < values_read; j++)
              chpl_ptr[i+j] = (int64_t)tmpArr[j];
            i+=values_read;
          }
          free(tmpArr);
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
          auto chpl_ptr = (bool*)chpl_arr;
          parquet::BoolReader* reader =
            static_cast<parquet::BoolReader*>(column_reader.get());

          while (reader->HasNext() && i < numElems) {
            if((numElems - i) < batchSize)
              batchSize = numElems - i;
            (void)reader->ReadBatch(batchSize, nullptr, nullptr, &chpl_ptr[i], &values_read);
            i+=values_read;
          }
        } else if(lty == ARROWFLOAT) {
          auto chpl_ptr = (double*)chpl_arr;
          parquet::FloatReader* reader =
            static_cast<parquet::FloatReader*>(column_reader.get());
          
          while (reader->HasNext() && i < numElems) {
            if((numElems - i) < batchSize) // adjust batchSize if needed
              batchSize = numElems - i;
            float* tmpArr = new float[batchSize]; // this will not include NaN values
            int16_t* def_lvl = new int16_t[batchSize];
            int16_t* rep_lvl = new int16_t[batchSize];
            (void)reader->ReadBatch(batchSize, def_lvl, rep_lvl, tmpArr, &values_read);

            int64_t tmp_offset = 0; // used to properly access tmp after NaN encountered
            int64_t val_idx = 0;
            for (int64_t j = 0; j< batchSize; j++){
              // skip any empty segments
              if (def_lvl[j] == 1)
                continue;
              
              // Null values treated as NaN
              if (def_lvl[j] == 2) {
                chpl_ptr[i+val_idx] = NAN;
                tmp_offset++; // adjustment for values array since Nulls are not included
              }
              else if (def_lvl[j] == 3){ // defined value to write
                chpl_ptr[i+val_idx] = (double)tmpArr[val_idx-tmp_offset];
              }
              val_idx++;
            }

            i += values_read + tmp_offset; // account for values and NaNs, but not empty segments

            delete[] tmpArr;
            delete[] def_lvl;
            delete[] rep_lvl;
          }
        } else if(lty == ARROWDOUBLE) {
          auto chpl_ptr = (double*)chpl_arr;
          parquet::DoubleReader* reader =
            static_cast<parquet::DoubleReader*>(column_reader.get());

          while (reader->HasNext() && i < numElems) {
            if((numElems - i) < batchSize) // adjust batchSize if needed
              batchSize = numElems - i;
            double* tmpArr = new double[batchSize]; // NaNs not included here
            int16_t* def_lvl = new int16_t[batchSize];
            int16_t* rep_lvl = new int16_t[batchSize];
            (void)reader->ReadBatch(batchSize, def_lvl, rep_lvl, tmpArr, &values_read);
            
            int64_t tmp_offset = 0; // used to properly access tmp after NaN encountered
            int64_t val_idx = 0;
            for (int64_t j = 0; j< batchSize; j++){
              // skip any empty segments
              if (def_lvl[j] == 1)
                continue;
              
              // Null values treated as NaN
              if (def_lvl[j] == 2) {
                chpl_ptr[i+val_idx] = NAN;
                tmp_offset++; // adjustment for values array since Nulls are not included
              }
              else if (def_lvl[j] == 3){ // defined value to write
                chpl_ptr[i+val_idx] = (double)tmpArr[val_idx-tmp_offset];
              }
              val_idx++;
            }

            i += values_read + tmp_offset; // account for values and NaNs, but not empty segments

            delete[] tmpArr;
            delete[] def_lvl;
            delete[] rep_lvl;
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

int cpp_readColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
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

      // Since int64 and uint64 Arrow dtypes share a physical type and only differ
      // in logical type, they must be read from the file in the same way
      if(ty == ARROWINT64 || ty == ARROWUINT64) {
        auto chpl_ptr = (int64_t*)chpl_arr;
        parquet::Int64Reader* reader =
          static_cast<parquet::Int64Reader*>(column_reader.get());
        startIdx -= reader->Skip(startIdx);

        while (reader->HasNext() && i < numElems) {
          if((numElems - i) < batchSize)
            batchSize = numElems - i;
          (void)reader->ReadBatch(batchSize, nullptr, nullptr, &chpl_ptr[i], &values_read);
          i+=values_read;
        }
      } else if(ty == ARROWINT32 || ty == ARROWUINT32) {
        auto chpl_ptr = (int64_t*)chpl_arr;
        parquet::Int32Reader* reader =
          static_cast<parquet::Int32Reader*>(column_reader.get());
        startIdx -= reader->Skip(startIdx);

        int32_t* tmpArr = (int32_t*)malloc(batchSize * sizeof(int32_t));
        while (reader->HasNext() && i < numElems) {
          if((numElems - i) < batchSize)
            batchSize = numElems - i;
          // Can't read directly into chpl_ptr because it is int64
          (void)reader->ReadBatch(batchSize, nullptr, nullptr, tmpArr, &values_read);
          for (int64_t j = 0; j < values_read; j++)
            chpl_ptr[i+j] = (int64_t)tmpArr[j];
          i+=values_read;
        }
        free(tmpArr);
      } else if(ty == ARROWBOOLEAN) {
        auto chpl_ptr = (bool*)chpl_arr;
        parquet::BoolReader* reader =
          static_cast<parquet::BoolReader*>(column_reader.get());
        startIdx -= reader->Skip(startIdx);

        while (reader->HasNext() && i < numElems) {
          if((numElems - i) < batchSize)
            batchSize = numElems - i;
          (void)reader->ReadBatch(batchSize, nullptr, nullptr, &chpl_ptr[i], &values_read);
          i+=values_read;
        }
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
        auto chpl_ptr = (double*)chpl_arr;
        parquet::FloatReader* reader =
          static_cast<parquet::FloatReader*>(column_reader.get());
        startIdx -= reader->Skip(startIdx);
        
        while (reader->HasNext() && i < numElems) {
          if((numElems - i) < batchSize) // adjust batchSize if needed
            batchSize = numElems - i;

          // define def and rep level tracking to the batch size. This is required to detect NaN
          int16_t* def_lvl = new int16_t[batchSize] { 0 };
          int16_t* rep_lvl = new int16_t[batchSize] { 0 };
          
          float* tmpArr = new float[batchSize] { 0 }; // this will not include NaN values
          
          int64_t idx_adjust = 0; // adjustment for NaNs encountered so index into tmpArr is correct
          (void)reader->ReadBatch(batchSize, def_lvl, rep_lvl, tmpArr, &values_read);
          // copy values to Chapel array. Convert to double if not NaN
          for (int64_t j = 0; j < batchSize; j++){
            // when definition level is 0, mean Null which equated to NaN here unless 0 is the max meaning no null/nan values
            if (max_def != 0 && def_lvl[j] == 0) {
              chpl_ptr[i] = NAN;
              idx_adjust++; // account for the NaN at the indexes after because tmpArr only stores values
            } else {
              chpl_ptr[i] = (double)tmpArr[j-idx_adjust]; // cast to double for Chapel
            }
            i++;
          }
          delete[] def_lvl;
          delete[] rep_lvl;
        }
      } else if(ty == ARROWDOUBLE) {
        auto chpl_ptr = (double*)chpl_arr;
        parquet::DoubleReader* reader =
          static_cast<parquet::DoubleReader*>(column_reader.get());
        startIdx -= reader->Skip(startIdx);

        while (reader->HasNext() && i < numElems) {
          if((numElems - i) < batchSize) // adjust batchSize if needed
            batchSize = numElems - i;

          // define def and rep level tracking to the batch size. This is required to detect NaN
          int16_t* def_lvl = new int16_t[batchSize] { 0 };
          int16_t* rep_lvl = new int16_t[batchSize] { 0 };

          double* tmpArr = new double[batchSize] { 0 }; // this will not include NaN values
          int64_t idx_adjust = 0; // adjustment for NaNs encountered so index into tmpArr is correct
          (void)reader->ReadBatch(batchSize, def_lvl, rep_lvl, tmpArr, &values_read);
          // copy values into our Chapel array
          for (int64_t j = 0; j < batchSize; j++){
            // when definition level is 0, mean Null which equated to NaN here unless 0 is the max meaning no null/nan values
            if (max_def != 0 && def_lvl[j] == 0) {
              chpl_ptr[i] = NAN;
              idx_adjust++; // account for the NaN at the indexes after because tmpArr only stores values
            } else {
              chpl_ptr[i] = tmpArr[j-idx_adjust];
            }
            i++;
          }
          delete[] tmpArr;
          delete[] def_lvl;
          delete[] rep_lvl;
        }
      } else if(ty == ARROWDECIMAL) {
        auto chpl_ptr = (double*)chpl_arr;
        parquet::FixedLenByteArray value;
        parquet::FixedLenByteArrayReader* reader =
          static_cast<parquet::FixedLenByteArrayReader*>(column_reader.get());
        startIdx -= reader->Skip(startIdx);

        while (reader->HasNext() && i < numElems) {
          if((numElems - i) < batchSize)
            batchSize = numElems - i;
          (void)reader->ReadBatch(1, nullptr, nullptr, &value, &values_read);
          arrow::Decimal128 v;
          PARQUET_ASSIGN_OR_THROW(v,
                                  ::arrow::Decimal128::FromBigEndian(value.ptr, 9));

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

// configure the schema for a multicolumn file
std::shared_ptr<parquet::schema::GroupNode> SetupSchema(void* column_names, void * objTypes, void* datatypes, int64_t colnum) {
  parquet::schema::NodeVector fields;
  auto cname_ptr = (char**)column_names;
  auto dtypes_ptr = (int64_t*) datatypes;
  auto objType_ptr = (int64_t*) objTypes;
  for (int64_t i = 0; i < colnum; i++){
    if(dtypes_ptr[i] == ARROWINT64) {
      if (objType_ptr[i] == SEGARRAY){
        auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::INT64, parquet::ConvertedType::NONE);
        auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
        fields.push_back(parquet::schema::GroupNode::Make(cname_ptr[i], parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
      } else {
        fields.push_back(parquet::schema::PrimitiveNode::Make(cname_ptr[i], parquet::Repetition::REQUIRED, parquet::Type::INT64, parquet::ConvertedType::NONE));
      }
    } else if(dtypes_ptr[i] == ARROWUINT64) {
      if (objType_ptr[i] == SEGARRAY){
        auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::INT64, parquet::ConvertedType::UINT_64);
        auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
        fields.push_back(parquet::schema::GroupNode::Make(cname_ptr[i], parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
      } else {
        fields.push_back(parquet::schema::PrimitiveNode::Make(cname_ptr[i], parquet::Repetition::REQUIRED, parquet::Type::INT64, parquet::ConvertedType::UINT_64));
      }
    } else if(dtypes_ptr[i] == ARROWBOOLEAN) {
      if (objType_ptr[i] == SEGARRAY){
        auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::BOOLEAN, parquet::ConvertedType::NONE);
        auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
        fields.push_back(parquet::schema::GroupNode::Make(cname_ptr[i], parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
      } else {
        fields.push_back(parquet::schema::PrimitiveNode::Make(cname_ptr[i], parquet::Repetition::REQUIRED, parquet::Type::BOOLEAN, parquet::ConvertedType::NONE));
      }
    } else if(dtypes_ptr[i] == ARROWDOUBLE) {
      if (objType_ptr[i] == SEGARRAY) {
        auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::DOUBLE, parquet::ConvertedType::NONE);
        auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
        fields.push_back(parquet::schema::GroupNode::Make(cname_ptr[i], parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
      } else {
        fields.push_back(parquet::schema::PrimitiveNode::Make(cname_ptr[i], parquet::Repetition::REQUIRED, parquet::Type::DOUBLE, parquet::ConvertedType::NONE));
      }
    } else if(dtypes_ptr[i] == ARROWSTRING) {
      if (objType_ptr[i] == SEGARRAY) {
        auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::BYTE_ARRAY, parquet::ConvertedType::NONE);
        auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
        fields.push_back(parquet::schema::GroupNode::Make(cname_ptr[i], parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
      } else {
        fields.push_back(parquet::schema::PrimitiveNode::Make(cname_ptr[i], parquet::Repetition::REQUIRED, parquet::Type::BYTE_ARRAY, parquet::ConvertedType::NONE));
      }
    }
  }
  return std::static_pointer_cast<parquet::schema::GroupNode>(
      parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));
}

int cpp_writeMultiColToParquet(const char* filename, void* column_names, 
                                void** ptr_arr, void** offset_arr, void* objTypes, void* datatypes,
                                void* segArr_sizes, int64_t colnum, int64_t numelems, int64_t rowGroupSize,
                                int64_t compression, char** errMsg) {
  try {
    // initialize the file to write to
    using FileClass = ::arrow::io::FileOutputStream;
    std::shared_ptr<FileClass> out_file;
    ARROWRESULT_OK(FileClass::Open(filename), out_file);

    // Setup the parquet schema
    std::shared_ptr<parquet::schema::GroupNode> schema = SetupSchema(column_names, objTypes, datatypes, colnum);

    parquet::WriterProperties::Builder builder;
    // assign the proper compression
    if(compression == SNAPPY_COMP) {
      builder.compression(parquet::Compression::SNAPPY);
    } else if (compression == GZIP_COMP) {
      builder.compression(parquet::Compression::GZIP);
    } else if (compression == BROTLI_COMP) {
      builder.compression(parquet::Compression::BROTLI);
    } else if (compression == ZSTD_COMP) {
      builder.compression(parquet::Compression::ZSTD);
    } else if (compression == LZ4_COMP) {
      builder.compression(parquet::Compression::LZ4);
    }
    std::shared_ptr<parquet::WriterProperties> props = builder.build();

    std::shared_ptr<parquet::ParquetFileWriter> file_writer =
      parquet::ParquetFileWriter::Open(out_file, schema, props);

    std::queue<int64_t> idxQueue_str; // queue used to track string byteIdx 
    std::queue<int64_t> idxQueue_segarray; // queue used to track index into the offsets

    auto dtypes_ptr = (int64_t*) datatypes;
    auto objType_ptr = (int64_t*) objTypes;
    auto saSizes_ptr = (int64_t*) segArr_sizes;
    int64_t numLeft = numelems; // number of elements remaining to write (rows)
    int64_t x = 0;  // index to start writing batch from
    while (numLeft > 0) {
      // Append a RowGroup with a specific number of rows.
      parquet::RowGroupWriter* rg_writer = file_writer->AppendRowGroup();
      int64_t batchSize = rowGroupSize;
      if(numLeft < rowGroupSize)
        batchSize = numLeft;

      // loop the columns and write the row groups
      for(int64_t i = 0; i < colnum; i++){
        int64_t dtype = dtypes_ptr[i];
        if (dtype == ARROWINT64 || dtype == ARROWUINT64) {
          auto data_ptr = (int64_t*)ptr_arr[i];
          parquet::Int64Writer* writer =
                static_cast<parquet::Int64Writer*>(rg_writer->NextColumn());

          if (objType_ptr[i] == SEGARRAY) {
            auto offset_ptr = (int64_t*)offset_arr[i];
            int64_t offIdx = 0; // index into offsets

            if (x > 0){
              offIdx = idxQueue_segarray.front();
              idxQueue_segarray.pop();
            }

            int64_t count = 0;
            while (count < batchSize) { // ensures rowGroupSize maintained
              int64_t segSize;
              if (offIdx == (numelems - 1)) {
                segSize = saSizes_ptr[i] - offset_ptr[offIdx];
              }
              else {
                segSize = offset_ptr[offIdx+1] - offset_ptr[offIdx];
              }
              if (segSize > 0) {
                int16_t* def_lvl = new int16_t[segSize] { 3 };
                int16_t* rep_lvl = new int16_t[segSize] { 0 };
                for (int64_t s = 0; s < segSize; s++){
                  // if the value is first in the segment rep_lvl = 0, otherwise 1
                  // all values defined at the item level (3)
                  rep_lvl[s] = (s == 0) ? 0 : 1;
                  def_lvl[s] = 3;
                }
                int64_t valIdx = offset_ptr[offIdx];
                writer->WriteBatch(segSize, def_lvl, rep_lvl, &data_ptr[valIdx]);
                delete[] def_lvl;
                delete[] rep_lvl;
              }
              else {
                // empty segment denoted by null value that is not repeated (first of segment) defined at the list level (1)
                segSize = 1; // even though segment is length=0, write null to hold the empty segment
                int16_t def_lvl = 1;
                int16_t rep_lvl = 0;
                writer->WriteBatch(segSize, &def_lvl, &rep_lvl, nullptr);
              }
              offIdx++;
              count++;
            }
            if (numLeft - count > 0) {
              idxQueue_segarray.push(offIdx);
            }
          } else {
            writer->WriteBatch(batchSize, nullptr, nullptr, &data_ptr[x]);
          }
        } else if(dtype == ARROWBOOLEAN) {
          auto data_ptr = (bool*)ptr_arr[i];
            parquet::BoolWriter* writer =
              static_cast<parquet::BoolWriter*>(rg_writer->NextColumn());
          if (objType_ptr[i] == SEGARRAY) {
            auto offset_ptr = (int64_t*)offset_arr[i];
            int64_t offIdx = 0; // index into offsets

            if (x > 0){
              offIdx = idxQueue_segarray.front();
              idxQueue_segarray.pop();
            }

            int64_t count = 0;
            while (count < batchSize) { // ensures rowGroupSize maintained
              int64_t segSize;
              if (offIdx == numelems - 1) {
                segSize = saSizes_ptr[i] - offset_ptr[offIdx];
              }
              else {
                segSize = offset_ptr[offIdx+1] - offset_ptr[offIdx];
              }
              if (segSize > 0) {
                int16_t* def_lvl = new int16_t[segSize] { 3 };
                int16_t* rep_lvl = new int16_t[segSize] { 0 };
                for (int64_t s = 0; s < segSize; s++){
                  // if the value is first in the segment rep_lvl = 0, otherwise 1
                  // all values defined at the item level (3)
                  rep_lvl[s] = (s == 0) ? 0 : 1;
                  def_lvl[s] = 3;
                }
                int64_t valIdx = offset_ptr[offIdx];
                writer->WriteBatch(segSize, def_lvl, rep_lvl, &data_ptr[valIdx]);
                delete[] def_lvl;
                delete[] rep_lvl;
              }
              else {
                // empty segment denoted by null value that is not repeated (first of segment) defined at the list level (1)
                segSize = 1; // even though segment is length=0, write null to hold the empty segment
                int16_t def_lvl = 1;
                int16_t rep_lvl = 0;
                writer->WriteBatch(segSize, &def_lvl, &rep_lvl, nullptr);
              }
              offIdx++;
              count++;
            }
            if (numLeft - count > 0) {
              idxQueue_segarray.push(offIdx);
            }
          } else {
            writer->WriteBatch(batchSize, nullptr, nullptr, &data_ptr[x]);
          }
        } else if(dtype == ARROWDOUBLE) {
          auto data_ptr = (double*)ptr_arr[i];
            parquet::DoubleWriter* writer =
          static_cast<parquet::DoubleWriter*>(rg_writer->NextColumn());
          if (objType_ptr[i] == SEGARRAY) {
            auto offset_ptr = (int64_t*)offset_arr[i];
            int64_t offIdx = 0; // index into offsets

            if (x > 0){
              offIdx = idxQueue_segarray.front();
              idxQueue_segarray.pop();
            }

            int64_t count = 0;
            while (count < batchSize) { // ensures rowGroupSize maintained
              int64_t segSize;
              if (offIdx == numelems - 1) {
                segSize = saSizes_ptr[i] - offset_ptr[offIdx];
              }
              else {
                segSize = offset_ptr[offIdx+1] - offset_ptr[offIdx];
              }
              if (segSize > 0) {
                int16_t* def_lvl = new int16_t[segSize] { 3 };
                int16_t* rep_lvl = new int16_t[segSize] { 0 };
                for (int64_t s = 0; s < segSize; s++){
                  // if the value is first in the segment rep_lvl = 0, otherwise 1
                  // all values defined at the item level (3)
                  rep_lvl[s] = (s == 0) ? 0 : 1;
                  def_lvl[s] = 3;
                }
                int64_t valIdx = offset_ptr[offIdx];
                writer->WriteBatch(segSize, def_lvl, rep_lvl, &data_ptr[valIdx]);
                delete[] def_lvl;
                delete[] rep_lvl;
              }
              else {
                // empty segment denoted by null value that is not repeated (first of segment) defined at the list level (1)
                segSize = 1; // even though segment is length=0, write null to hold the empty segment
                int16_t def_lvl = 1;
                int16_t rep_lvl =0;
                writer->WriteBatch(segSize, &def_lvl, &rep_lvl, nullptr);
              }
              offIdx++;
              count++;
            }
            if (numLeft - count > 0) {
              idxQueue_segarray.push(offIdx);
            }
          } else {
            writer->WriteBatch(batchSize, nullptr, nullptr, &data_ptr[x]);
          }
        } else if(dtype == ARROWSTRING) {
          auto data_ptr = (uint8_t*)ptr_arr[i];
          parquet::ByteArrayWriter* ba_writer =
          static_cast<parquet::ByteArrayWriter*>(rg_writer->NextColumn());
          if (objType_ptr[i] == SEGARRAY) {
            auto offset_ptr = (int64_t*)offset_arr[i];
            int64_t byteIdx = 0;
            int64_t offIdx = 0; // index into offsets

            // identify the starting byte index
            if (x > 0){
              byteIdx = idxQueue_str.front();
              idxQueue_str.pop();

              offIdx = idxQueue_segarray.front();
              idxQueue_segarray.pop();
            }

            int64_t count = 0;
            while (count < batchSize) { // ensures rowGroupSize maintained
              int64_t segSize;
              if (offIdx == numelems - 1) {
                segSize = saSizes_ptr[i] - offset_ptr[offIdx];
              }
              else {
                segSize = offset_ptr[offIdx+1] - offset_ptr[offIdx];
              }
              if (segSize > 0) {
                for (int64_t s=0; s<segSize; s++) {
                  int16_t def_lvl = 3;
                  int16_t rep_lvl = (s == 0) ? 0 : 1;
                  parquet::ByteArray value;
                  value.ptr = reinterpret_cast<const uint8_t*>(&data_ptr[byteIdx]);
                  int64_t nextIdx = byteIdx;
                  while (data_ptr[nextIdx] != 0x00){
                    nextIdx++;
                  }
                  value.len = nextIdx - byteIdx;
                  ba_writer->WriteBatch(1, &def_lvl, &rep_lvl, &value);
                  byteIdx = nextIdx + 1; // increment to start of next word
                }
              }
              else {
                // empty segment denoted by null value that is not repeated (first of segment) defined at the list level (1)
                segSize = 1; // even though segment is length=0, write null to hold the empty segment
                int16_t* def_lvl = new int16_t[segSize] { 1 };
                int16_t* rep_lvl = new int16_t[segSize] { 0 };
                ba_writer->WriteBatch(segSize, def_lvl, rep_lvl, nullptr);
              }
              offIdx++;
              count++;
            }
            if (numLeft - count > 0) {
              idxQueue_str.push(byteIdx);
              idxQueue_segarray.push(offIdx);
            }
          }
          else {
            int64_t count = 0;
            int64_t byteIdx = 0;

            // identify the starting byte index
            if (x > 0){
              byteIdx = idxQueue_str.front();
              idxQueue_str.pop();
            }
            
            while(count < batchSize) {
              parquet::ByteArray value;
              int16_t definition_level = 1;
              value.ptr = reinterpret_cast<const uint8_t*>(&data_ptr[byteIdx]);
              int64_t nextIdx = byteIdx;
              while (data_ptr[nextIdx] != 0x00){
                nextIdx++;
              }
              // subtract 1 since we have the null terminator
              value.len = nextIdx - byteIdx;
              ba_writer->WriteBatch(1, &definition_level, nullptr, &value);
              count++;
              byteIdx = nextIdx + 1;
            }
            if (numLeft - count > 0) {
              idxQueue_str.push(byteIdx);
            }
          }
        } else {
          return ARROWERROR;
        }
      }
      numLeft -= batchSize;
      x += batchSize;
    }

    file_writer->Close();
    ARROWSTATUS_OK(out_file->Close());
    
    return 0;
   } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int cpp_writeColumnToParquet(const char* filename, void* chpl_arr,
                             int64_t colnum, const char* dsetname, int64_t numelems,
                             int64_t rowGroupSize, int64_t dtype, int64_t compression,
                             char** errMsg) {
  try {
    using FileClass = ::arrow::io::FileOutputStream;
    std::shared_ptr<FileClass> out_file;
    ARROWRESULT_OK(FileClass::Open(filename), out_file);

    parquet::schema::NodeVector fields;
    if(dtype == ARROWINT64)
      fields.push_back(parquet::schema::PrimitiveNode::Make(dsetname, parquet::Repetition::REQUIRED, parquet::Type::INT64, parquet::ConvertedType::NONE));
    else if(dtype == ARROWUINT64)
      fields.push_back(parquet::schema::PrimitiveNode::Make(dsetname, parquet::Repetition::REQUIRED, parquet::Type::INT64, parquet::ConvertedType::UINT_64));
    else if(dtype == ARROWBOOLEAN)
      fields.push_back(parquet::schema::PrimitiveNode::Make(dsetname, parquet::Repetition::REQUIRED, parquet::Type::BOOLEAN, parquet::ConvertedType::NONE));
    else if(dtype == ARROWDOUBLE)
      fields.push_back(parquet::schema::PrimitiveNode::Make(dsetname, parquet::Repetition::REQUIRED, parquet::Type::DOUBLE, parquet::ConvertedType::NONE));
    std::shared_ptr<parquet::schema::GroupNode> schema = std::static_pointer_cast<parquet::schema::GroupNode>
      (parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));

    parquet::WriterProperties::Builder builder;
    // assign the proper compression
    if(compression == SNAPPY_COMP) {
      builder.compression(parquet::Compression::SNAPPY);
    } else if (compression == GZIP_COMP) {
      builder.compression(parquet::Compression::GZIP);
    } else if (compression == BROTLI_COMP) {
      builder.compression(parquet::Compression::BROTLI);
    } else if (compression == ZSTD_COMP) {
      builder.compression(parquet::Compression::ZSTD);
    } else if (compression == LZ4_COMP) {
      builder.compression(parquet::Compression::LZ4);
    }
    std::shared_ptr<parquet::WriterProperties> props = builder.build();

    std::shared_ptr<parquet::ParquetFileWriter> file_writer =
      parquet::ParquetFileWriter::Open(out_file, schema, props);

    int64_t i = 0;
    int64_t numLeft = numelems;

    if (chpl_arr == NULL) {
      // early out to prevent bad memory access
      return 0;
    }

    if(dtype == ARROWINT64 || dtype == ARROWUINT64) {
      auto chpl_ptr = (int64_t*)chpl_arr;
      while(numLeft > 0) {
        parquet::RowGroupWriter* rg_writer = file_writer->AppendRowGroup();
        parquet::Int64Writer* int64_writer =
          static_cast<parquet::Int64Writer*>(rg_writer->NextColumn());

        int64_t batchSize = rowGroupSize;
        if(numLeft < rowGroupSize)
          batchSize = numLeft;
        int64_writer->WriteBatch(batchSize, nullptr, nullptr, &chpl_ptr[i]);
        numLeft -= batchSize;
        i += batchSize;
      }
    } else if(dtype == ARROWBOOLEAN) {
      auto chpl_ptr = (bool*)chpl_arr;
      while(numLeft > 0) {
        parquet::RowGroupWriter* rg_writer = file_writer->AppendRowGroup();
        parquet::BoolWriter* writer =
          static_cast<parquet::BoolWriter*>(rg_writer->NextColumn());

        int64_t batchSize = rowGroupSize;
        if(numLeft < rowGroupSize)
          batchSize = numLeft;
        writer->WriteBatch(batchSize, nullptr, nullptr, &chpl_ptr[i]);
        numLeft -= batchSize;
        i += batchSize;
      }
    } else if(dtype == ARROWDOUBLE) {
      auto chpl_ptr = (double*)chpl_arr;
      while(numLeft > 0) {
        parquet::RowGroupWriter* rg_writer = file_writer->AppendRowGroup();
        parquet::DoubleWriter* writer =
          static_cast<parquet::DoubleWriter*>(rg_writer->NextColumn());

        int64_t batchSize = rowGroupSize;
        if(numLeft < rowGroupSize)
          batchSize = numLeft;
        writer->WriteBatch(batchSize, nullptr, nullptr, &chpl_ptr[i]);
        numLeft -= batchSize;
        i += batchSize;
      }
    } else {
      return ARROWERROR;
    }

    file_writer->Close();
    ARROWSTATUS_OK(out_file->Close());

    return 0;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int cpp_writeStrColumnToParquet(const char* filename, void* chpl_arr, void* chpl_offsets,
                                const char* dsetname, int64_t numelems,
                                int64_t rowGroupSize, int64_t dtype, int64_t compression,
                                char** errMsg) {
  try {
    using FileClass = ::arrow::io::FileOutputStream;
    std::shared_ptr<FileClass> out_file;
    PARQUET_ASSIGN_OR_THROW(out_file, FileClass::Open(filename));

    parquet::schema::NodeVector fields;

    fields.push_back(parquet::schema::PrimitiveNode::Make(dsetname, parquet::Repetition::OPTIONAL, parquet::Type::BYTE_ARRAY, parquet::ConvertedType::NONE));
    std::shared_ptr<parquet::schema::GroupNode> schema = std::static_pointer_cast<parquet::schema::GroupNode>
      (parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));

    parquet::WriterProperties::Builder builder;
    // assign the proper compression
    if(compression == SNAPPY_COMP) {
      builder.compression(parquet::Compression::SNAPPY);
    } else if (compression == GZIP_COMP) {
      builder.compression(parquet::Compression::GZIP);
    } else if (compression == BROTLI_COMP) {
      builder.compression(parquet::Compression::BROTLI);
    } else if (compression == ZSTD_COMP) {
      builder.compression(parquet::Compression::ZSTD);
    } else if (compression == LZ4_COMP) {
      builder.compression(parquet::Compression::LZ4);
    }
    std::shared_ptr<parquet::WriterProperties> props = builder.build();

    std::shared_ptr<parquet::ParquetFileWriter> file_writer =
      parquet::ParquetFileWriter::Open(out_file, schema, props);

    int64_t i = 0;
    int64_t numLeft = numelems;

    if(dtype == ARROWSTRING) {
      auto chpl_ptr = (uint8_t*)chpl_arr;
      auto offsets = (int64_t*)chpl_offsets;
      int64_t byteIdx = 0;
      int64_t offIdx = 0;
      
      while(numLeft > 0) {
        parquet::RowGroupWriter* rg_writer = file_writer->AppendRowGroup();
        parquet::ByteArrayWriter* ba_writer =
          static_cast<parquet::ByteArrayWriter*>(rg_writer->NextColumn());
        int64_t count = 0;
        while(numLeft > 0 && count < rowGroupSize) {
          parquet::ByteArray value;
          int16_t definition_level = 1;
          value.ptr = reinterpret_cast<const uint8_t*>(&chpl_ptr[byteIdx]);
          // subtract 1 since we have the null terminator
          value.len = offsets[offIdx+1] - offsets[offIdx] - 1;
          if (value.len == 0)
            definition_level = 0;
          ba_writer->WriteBatch(1, &definition_level, nullptr, &value);
          numLeft--;count++;
          offIdx++;
          byteIdx+=offsets[offIdx] - offsets[offIdx-1];
        }
      }
    } else {
      return ARROWERROR;
    }

    file_writer->Close();
    ARROWSTATUS_OK(out_file->Close());

    return 0;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int cpp_writeStrListColumnToParquet(const char* filename, void* chpl_segs, void* chpl_offsets, void* chpl_arr,
                                const char* dsetname, int64_t numelems,
                                int64_t rowGroupSize, int64_t dtype, int64_t compression,
                                char** errMsg) {
  try {
    if(dtype == ARROWSTRING) { // check the type here so if it is wrong we don't create a bad file
      using FileClass = ::arrow::io::FileOutputStream;
      std::shared_ptr<FileClass> out_file;
      PARQUET_ASSIGN_OR_THROW(out_file, FileClass::Open(filename));

      parquet::schema::NodeVector fields;

      auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::BYTE_ARRAY, parquet::ConvertedType::NONE);
      auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
      fields.push_back(parquet::schema::GroupNode::Make(dsetname, parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
      std::shared_ptr<parquet::schema::GroupNode> schema = std::static_pointer_cast<parquet::schema::GroupNode>
          (parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));

      parquet::WriterProperties::Builder builder;
      // assign the proper compression
      if(compression == SNAPPY_COMP) {
        builder.compression(parquet::Compression::SNAPPY);
      } else if (compression == GZIP_COMP) {
        builder.compression(parquet::Compression::GZIP);
      } else if (compression == BROTLI_COMP) {
        builder.compression(parquet::Compression::BROTLI);
      } else if (compression == ZSTD_COMP) {
        builder.compression(parquet::Compression::ZSTD);
      } else if (compression == LZ4_COMP) {
        builder.compression(parquet::Compression::LZ4);
      }
      std::shared_ptr<parquet::WriterProperties> props = builder.build();

      std::shared_ptr<parquet::ParquetFileWriter> file_writer =
        parquet::ParquetFileWriter::Open(out_file, schema, props);

      int64_t i = 0;
      int64_t numLeft = numelems;
      auto segments = (int64_t*)chpl_segs;
      int64_t segIdx = 0; // index into segarray segments
      int64_t offIdx = 0; // index into the segstring segments
      int64_t valIdx = 0; // index into chpl_arr

      while(numLeft > 0) { // write all local values to the file
        parquet::RowGroupWriter* rg_writer = file_writer->AppendRowGroup();
        parquet::ByteArrayWriter* ba_writer =
          static_cast<parquet::ByteArrayWriter*>(rg_writer->NextColumn());
        int64_t count = 0;
        while (numLeft > 0 && count < rowGroupSize) { // ensures rowGroupSize maintained
          int64_t segmentLength = segments[segIdx+1] - segments[segIdx];
          if (segmentLength > 0) {
            auto offsets = (int64_t*)chpl_offsets;
            auto chpl_ptr = (uint8_t*)chpl_arr;
            for (int64_t x = 0; x < segmentLength; x++){
              int16_t rep_lvl = (x == 0) ? 0 : 1;
              int16_t def_lvl = 3;
              parquet::ByteArray value;
              value.ptr = reinterpret_cast<const uint8_t*>(&chpl_ptr[valIdx]);
              value.len = offsets[offIdx+1] - offsets[offIdx] - 1;
              ba_writer->WriteBatch(1, &def_lvl, &rep_lvl, &value);
              offIdx++;
              valIdx+=offsets[offIdx] - offsets[offIdx-1];
            }
          } else {
            // empty segment denoted by null value that is not repeated (first of segment) defined at the list level (1)
            segmentLength = 1; // even though segment is length=0, write null to hold the empty segment
            int16_t def_lvl = 1;
            int16_t rep_lvl = 0;
            ba_writer->WriteBatch(segmentLength, &def_lvl, &rep_lvl, nullptr);
          }
          segIdx++;
          numLeft--;count++;
        }
      }

      file_writer->Close();
      ARROWSTATUS_OK(out_file->Close());
      return 0;
    } else {
      return ARROWERROR;
    }
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int cpp_writeListColumnToParquet(const char* filename, void* chpl_segs, void* chpl_arr,
                                const char* dsetname, int64_t numelems,
                                int64_t rowGroupSize, int64_t dtype, int64_t compression,
                                char** errMsg) {
  try {
    using FileClass = ::arrow::io::FileOutputStream;
    std::shared_ptr<FileClass> out_file;
    PARQUET_ASSIGN_OR_THROW(out_file, FileClass::Open(filename));

    parquet::schema::NodeVector fields;

    // create the list schema. List containing the dtype
    if (dtype == ARROWINT64) {
      auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::INT64, parquet::ConvertedType::NONE);
      auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
      fields.push_back(parquet::schema::GroupNode::Make(dsetname, parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
    }
    else if (dtype == ARROWUINT64) {
      auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::INT64, parquet::ConvertedType::UINT_64);
      auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
      fields.push_back(parquet::schema::GroupNode::Make(dsetname, parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
    }
    else if (dtype == ARROWBOOLEAN) {
      auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::BOOLEAN, parquet::ConvertedType::NONE);
      auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
      fields.push_back(parquet::schema::GroupNode::Make(dsetname, parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
    }
    else if (dtype == ARROWDOUBLE) {
      auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::DOUBLE, parquet::ConvertedType::NONE);
      auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
      fields.push_back(parquet::schema::GroupNode::Make(dsetname, parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
    }
    std::shared_ptr<parquet::schema::GroupNode> schema = std::static_pointer_cast<parquet::schema::GroupNode>
      (parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));

    parquet::WriterProperties::Builder builder;
    // assign the proper compression
    if(compression == SNAPPY_COMP) {
      builder.compression(parquet::Compression::SNAPPY);
    } else if (compression == GZIP_COMP) {
      builder.compression(parquet::Compression::GZIP);
    } else if (compression == BROTLI_COMP) {
      builder.compression(parquet::Compression::BROTLI);
    } else if (compression == ZSTD_COMP) {
      builder.compression(parquet::Compression::ZSTD);
    } else if (compression == LZ4_COMP) {
      builder.compression(parquet::Compression::LZ4);
    }
    std::shared_ptr<parquet::WriterProperties> props = builder.build();

    std::shared_ptr<parquet::ParquetFileWriter> file_writer =
      parquet::ParquetFileWriter::Open(out_file, schema, props);

    int64_t i = 0;
    int64_t numLeft = numelems;
    auto segments = (int64_t*)chpl_segs;
    int64_t valIdx = 0; // index into chpl_arr
    int64_t segIdx = 0; // index into offsets

    if(dtype == ARROWINT64 || dtype == ARROWUINT64) {     
      while(numLeft > 0) { // write all local values to the file
        parquet::RowGroupWriter* rg_writer = file_writer->AppendRowGroup();
        parquet::Int64Writer* writer =
          static_cast<parquet::Int64Writer*>(rg_writer->NextColumn());
        int64_t count = 0;
        while (numLeft > 0 && count < rowGroupSize) { // ensures rowGroupSize maintained
          int64_t batchSize = segments[segIdx+1] - segments[segIdx];
          if (batchSize > 0) {
            auto chpl_ptr = (int64_t*)chpl_arr;
            int16_t* def_lvl = new int16_t[batchSize] { 3 }; // all values defined at the item level (3)
            int16_t* rep_lvl = new int16_t[batchSize] { 0 };
            for (int64_t x = 0; x < batchSize; x++){
              // if the value is first in the segment rep_lvl = 0, otherwise 1
              rep_lvl[x] = (x == 0) ? 0 : 1;
              def_lvl[x] = 3;
            }
            writer->WriteBatch(batchSize, def_lvl, rep_lvl, &chpl_ptr[valIdx]);
            valIdx += batchSize;
            delete[] def_lvl;
            delete[] rep_lvl;
          }
          else {
            // empty segment denoted by null value that is not repeated (first of segment) defined at the list level (1)
            batchSize = 1; // even though segment is length=0, write null to hold the empty segment
            int16_t def_lvl = 1;
            int16_t rep_lvl = 0;
            writer->WriteBatch(batchSize, &def_lvl, &rep_lvl, nullptr);
          }
          count++;
          segIdx++;
          numLeft--;
        }
      }
    }
    else if (dtype == ARROWBOOLEAN) {
      while(numLeft > 0) {
        parquet::RowGroupWriter* rg_writer = file_writer->AppendRowGroup();
        parquet::BoolWriter* writer =
          static_cast<parquet::BoolWriter*>(rg_writer->NextColumn());
        int64_t count = 0;
        while (numLeft > 0 && count < rowGroupSize) {
          int64_t batchSize = segments[segIdx+1] - segments[segIdx];
          if (batchSize > 0) {
            auto chpl_ptr = (bool*)chpl_arr;
            // if the value is first in the segment rep_lvl = 0, otherwise 1
            // all values defined at the item level (3)
            int16_t* def_lvl = new int16_t[batchSize] { 3 };
            int16_t* rep_lvl = new int16_t[batchSize] { 0 };
            for (int64_t x = 0; x < batchSize; x++){
              rep_lvl[x] = (x == 0) ? 0 : 1;
              def_lvl[x] = 3;
            }
            writer->WriteBatch(batchSize, def_lvl, rep_lvl, &chpl_ptr[valIdx]);
            valIdx += batchSize;
            delete[] def_lvl;
            delete[] rep_lvl;
          }
          else {
            // empty segment denoted by null value that is not repeated (first of segment) defined at the list level (1)
            batchSize = 1; // even though segment is length=0, write null to hold the empty segment
            int16_t def_lvl = 1;
            int16_t rep_lvl = 0;
            writer->WriteBatch(batchSize, &def_lvl, &rep_lvl, nullptr);
          }
          count++;
          segIdx++;
          numLeft--;
        }
      }
    }
    else if (dtype == ARROWDOUBLE) {      
      while(numLeft > 0) {
        parquet::RowGroupWriter* rg_writer = file_writer->AppendRowGroup();
        parquet::DoubleWriter* writer =
          static_cast<parquet::DoubleWriter*>(rg_writer->NextColumn());
        int64_t count = 0;
        while (numLeft > 0 && count < rowGroupSize) {
          int64_t batchSize = segments[segIdx+1] - segments[segIdx];
          if (batchSize > 0) {
            auto chpl_ptr = (double*)chpl_arr;
            // if the value is first in the segment rep_lvl = 0, otherwise 1
            // all values defined at the item level (3)
            int16_t* def_lvl = new int16_t[batchSize] { 3 };
            int16_t* rep_lvl = new int16_t[batchSize] { 0 };
            for (int64_t x = 0; x < batchSize; x++){
              rep_lvl[x] = (x == 0) ? 0 : 1;
              def_lvl[x] = 3;
            }
            writer->WriteBatch(batchSize, def_lvl, rep_lvl, &chpl_ptr[valIdx]);
            valIdx += batchSize;
            delete[] def_lvl;
            delete[] rep_lvl;
          }
          else {
            // empty segment denoted by null value that is not repeated (first of segment) defined at the list level (1)
            batchSize = 1; // even though segment is length=0, write null to hold the empty segment
            int16_t def_lvl = 1;
            int16_t rep_lvl = 0;
            writer->WriteBatch(batchSize, &def_lvl, &rep_lvl, nullptr);
          }
          count++;
          segIdx++;
          numLeft--;
        }
      }
    }
    else {
      return ARROWERROR;
    }

    file_writer->Close();
    ARROWSTATUS_OK(out_file->Close());

    return 0;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int cpp_createEmptyListParquetFile(const char* filename, const char* dsetname, int64_t dtype,
                               int64_t compression, char** errMsg) {
  try {
    using FileClass = ::arrow::io::FileOutputStream;
    std::shared_ptr<FileClass> out_file;
    PARQUET_ASSIGN_OR_THROW(out_file, FileClass::Open(filename));

    parquet::schema::NodeVector fields;
    if (dtype == ARROWINT64) {
      auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::INT64, parquet::ConvertedType::NONE);
      auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
      fields.push_back(parquet::schema::GroupNode::Make(dsetname, parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
    }
    else if (dtype == ARROWUINT64) {
      auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::INT64, parquet::ConvertedType::UINT_64);
      auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
      fields.push_back(parquet::schema::GroupNode::Make(dsetname, parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
    }
    else if (dtype == ARROWBOOLEAN) {
      auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::BOOLEAN, parquet::ConvertedType::NONE);
      auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
      fields.push_back(parquet::schema::GroupNode::Make(dsetname, parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
    }
    else if (dtype == ARROWDOUBLE) {
      auto element = parquet::schema::PrimitiveNode::Make("item", parquet::Repetition::OPTIONAL, parquet::Type::DOUBLE, parquet::ConvertedType::NONE);
      auto list = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED, {element});
      fields.push_back(parquet::schema::GroupNode::Make(dsetname, parquet::Repetition::OPTIONAL, {list}, parquet::ConvertedType::LIST));
    }
    std::shared_ptr<parquet::schema::GroupNode> schema = std::static_pointer_cast<parquet::schema::GroupNode>
      (parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));

    parquet::WriterProperties::Builder builder;
    // assign the proper compression
    if(compression == SNAPPY_COMP) {
      builder.compression(parquet::Compression::SNAPPY);
    } else if (compression == GZIP_COMP) {
      builder.compression(parquet::Compression::GZIP);
    } else if (compression == BROTLI_COMP) {
      builder.compression(parquet::Compression::BROTLI);
    } else if (compression == ZSTD_COMP) {
      builder.compression(parquet::Compression::ZSTD);
    } else if (compression == LZ4_COMP) {
      builder.compression(parquet::Compression::LZ4);
    }
    std::shared_ptr<parquet::WriterProperties> props = builder.build();
    std::shared_ptr<parquet::ParquetFileWriter> file_writer =
      parquet::ParquetFileWriter::Open(out_file, schema, props);

    file_writer->Close();
    ARROWSTATUS_OK(out_file->Close());

    return 0;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int cpp_createEmptyParquetFile(const char* filename, const char* dsetname, int64_t dtype,
                               int64_t compression, char** errMsg) {
  try {
    using FileClass = ::arrow::io::FileOutputStream;
    std::shared_ptr<FileClass> out_file;
    PARQUET_ASSIGN_OR_THROW(out_file, FileClass::Open(filename));

    parquet::schema::NodeVector fields;
    if(dtype == ARROWINT64)
      fields.push_back(parquet::schema::PrimitiveNode::Make(dsetname, parquet::Repetition::REQUIRED, parquet::Type::INT64, parquet::ConvertedType::NONE));
    else if(dtype == ARROWUINT64)
      fields.push_back(parquet::schema::PrimitiveNode::Make(dsetname, parquet::Repetition::REQUIRED, parquet::Type::INT64, parquet::ConvertedType::UINT_64));
    else if(dtype == ARROWBOOLEAN)
      fields.push_back(parquet::schema::PrimitiveNode::Make(dsetname, parquet::Repetition::REQUIRED, parquet::Type::BOOLEAN, parquet::ConvertedType::NONE));
    else if(dtype == ARROWDOUBLE)
      fields.push_back(parquet::schema::PrimitiveNode::Make(dsetname, parquet::Repetition::REQUIRED, parquet::Type::DOUBLE, parquet::ConvertedType::NONE));
    else if(dtype == ARROWSTRING)
      fields.push_back(parquet::schema::PrimitiveNode::Make(dsetname, parquet::Repetition::OPTIONAL, parquet::Type::BYTE_ARRAY, parquet::ConvertedType::NONE));
    std::shared_ptr<parquet::schema::GroupNode> schema = std::static_pointer_cast<parquet::schema::GroupNode>
      (parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));

    parquet::WriterProperties::Builder builder;
    // assign the proper compression
    if(compression == SNAPPY_COMP) {
      builder.compression(parquet::Compression::SNAPPY);
    } else if (compression == GZIP_COMP) {
      builder.compression(parquet::Compression::GZIP);
    } else if (compression == BROTLI_COMP) {
      builder.compression(parquet::Compression::BROTLI);
    } else if (compression == ZSTD_COMP) {
      builder.compression(parquet::Compression::ZSTD);
    } else if (compression == LZ4_COMP) {
      builder.compression(parquet::Compression::LZ4);
    }
    std::shared_ptr<parquet::WriterProperties> props = builder.build();
    std::shared_ptr<parquet::ParquetFileWriter> file_writer =
      parquet::ParquetFileWriter::Open(out_file, schema, props);

    file_writer->Close();
    ARROWSTATUS_OK(out_file->Close());

    return 0;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

int cpp_appendColumnToParquet(const char* filename, void* chpl_arr,
                              const char* dsetname, int64_t numelems,
                              int64_t dtype, int64_t compression,
                              char** errMsg) {
  try {
    if (chpl_arr == NULL){
      // early out to prevent bad memory access
      return 0;
    }
    std::shared_ptr<arrow::io::ReadableFile> infile;
    ARROWRESULT_OK(arrow::io::ReadableFile::Open(filename, arrow::default_memory_pool()),
                   infile);
    std::unique_ptr<parquet::arrow::FileReader> reader;
    ARROWSTATUS_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
    // Use threads for case when reading a table with many columns
    reader->set_use_threads(true);

    std::shared_ptr<arrow::Table> table;
    std::shared_ptr<arrow::Table>* hold_table = &table;
    ARROWSTATUS_OK(reader->ReadTable(hold_table));

    arrow::ArrayVector arrays;
    std::shared_ptr<arrow::Array> values;
    auto chunk_type = arrow::int64();
    if(dtype == ARROWINT64) {
      chunk_type = arrow::int64();
      arrow::Int64Builder builder;
      auto chpl_ptr = (int64_t*)chpl_arr;
      ARROWSTATUS_OK(builder.AppendValues(chpl_ptr, numelems, nullptr))
      ARROWSTATUS_OK(builder.Finish(&values));
    } else if(dtype == ARROWUINT64) {
      chunk_type = arrow::uint64();
      arrow::UInt64Builder builder;
      auto chpl_ptr = (uint64_t*)chpl_arr;
      ARROWSTATUS_OK(builder.AppendValues(chpl_ptr, numelems, nullptr))
      ARROWSTATUS_OK(builder.Finish(&values));
    } else if(dtype == ARROWBOOLEAN) {
      chunk_type = arrow::boolean();
      arrow::BooleanBuilder builder;
      auto chpl_ptr = (uint8_t*)chpl_arr;
      ARROWSTATUS_OK(builder.AppendValues(chpl_ptr, numelems, nullptr))
      ARROWSTATUS_OK(builder.Finish(&values));
    } else if(dtype == ARROWSTRING) {
      chunk_type = arrow::utf8();
      arrow::StringBuilder builder;
      auto chpl_ptr = (uint8_t*)chpl_arr;
      int64_t j = 0;
      for(int64_t i = 0; i < numelems; i++) {
        std::string tmp_str = "";
        while(chpl_ptr[j] != 0x00) {
          tmp_str += chpl_ptr[j++];
        }
        j++;
        
        auto const status = builder.Append(tmp_str);
        if (status.IsCapacityError()) {
          // Reached current chunk's capacity limit, so start a new one...
          ARROWSTATUS_OK(builder.Finish(&values));
          arrays.push_back(values);
          values.reset();
          builder.Reset();
          
          // ...with this string as its first item.
          ARROWSTATUS_OK(builder.Append(tmp_str));
        } else {
          ARROWSTATUS_OK(status);
        }
      }
      ARROWSTATUS_OK(builder.Finish(&values));
    } else if(dtype == ARROWDOUBLE) {
      chunk_type = arrow::float64();
      arrow::DoubleBuilder builder;
      auto chpl_ptr = (double*)chpl_arr;
      ARROWSTATUS_OK(builder.AppendValues(chpl_ptr, numelems, nullptr))
      ARROWSTATUS_OK(builder.Finish(&values));
    } else {
      std::string msg = "Unrecognized Parquet dtype"; 
      *errMsg = strdup(msg.c_str());
      return ARROWERROR;
    }
    arrays.push_back(values);

    std::shared_ptr<arrow::ChunkedArray> chunk_sh_ptr;
    ARROWRESULT_OK(arrow::ChunkedArray::Make({arrays}, chunk_type), chunk_sh_ptr);

    auto newField = arrow::field(dsetname, chunk_type);
    std::shared_ptr<arrow::Table> fin_table;
    ARROWRESULT_OK(table -> AddColumn(0, newField, chunk_sh_ptr), fin_table);

    using FileClass = ::arrow::io::FileOutputStream;
    std::shared_ptr<FileClass> out_file;
    ARROWRESULT_OK(FileClass::Open(filename), out_file);
    ARROWSTATUS_OK(parquet::arrow::WriteTable(*fin_table, arrow::default_memory_pool(), out_file, numelems));
    
    return 0;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
}

const char* cpp_getVersionInfo(void) {
  return strdup(arrow::GetBuildInfo().version_string.c_str());
}

int cpp_getDatasetNames(const char* filename, char** dsetResult, bool readNested, char** errMsg) {
  try {
    std::shared_ptr<arrow::io::ReadableFile> infile;
    ARROWRESULT_OK(arrow::io::ReadableFile::Open(filename, arrow::default_memory_pool()),
                   infile);
    std::unique_ptr<parquet::arrow::FileReader> reader;
    ARROWSTATUS_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

    std::shared_ptr<arrow::Schema> sc;
    std::shared_ptr<arrow::Schema>* out = &sc;
    ARROWSTATUS_OK(reader->GetSchema(out));

    std::string fields = "";
    bool first = true;

    for(int i = 0; i < sc->num_fields(); i++) {
      // only add fields of supported types
      if(sc->field(i)->type()->id() == arrow::Type::INT64 ||
         sc->field(i)->type()->id() == arrow::Type::INT32 ||
         sc->field(i)->type()->id() == arrow::Type::INT16 ||
         sc->field(i)->type()->id() == arrow::Type::UINT64 ||
         sc->field(i)->type()->id() == arrow::Type::UINT32 ||
         sc->field(i)->type()->id() == arrow::Type::UINT16 ||
         sc->field(i)->type()->id() == arrow::Type::TIMESTAMP ||
         sc->field(i)->type()->id() == arrow::Type::BOOL ||
         sc->field(i)->type()->id() == arrow::Type::STRING ||
         sc->field(i)->type()->id() == arrow::Type::BINARY ||
         sc->field(i)->type()->id() == arrow::Type::FLOAT ||
         sc->field(i)->type()->id() == arrow::Type::DOUBLE ||
         (sc->field(i)->type()->id() == arrow::Type::LIST && readNested) ||
         sc->field(i)->type()->id() == arrow::Type::DECIMAL
         ) {
        if(!first)
          fields += ("," + sc->field(i)->name());
        else
          fields += (sc->field(i)->name());
        first = false;
      } else if (sc->field(i)->type()->id() == arrow::Type::LIST && !readNested) {
        continue;
      } else {
        std::string fname(filename);
        std::string dname(sc->field(i)->ToString());
        std::string msg = "Unsupported type on column: " + dname + " in " + filename; 
        *errMsg = strdup(msg.c_str());
        return ARROWERROR;
      }
    }
    *dsetResult = strdup(fields.c_str());
  
    return 0;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
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
  int64_t c_getNumRows(const char* chpl_str, char** errMsg) {
    return cpp_getNumRows(chpl_str, errMsg);
  }

  int c_readListColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
    return cpp_readListColumnByName(filename, chpl_arr, colname, numElems, startIdx, batchSize, errMsg);
  }

  int c_readColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
    return cpp_readColumnByName(filename, chpl_arr, colname, numElems, startIdx, batchSize, errMsg);
  }

  int c_getType(const char* filename, const char* colname, char** errMsg) {
    return cpp_getType(filename, colname, errMsg);
  }

  int c_getListType(const char* filename, const char* colname, char** errMsg) {
    return cpp_getListType(filename, colname, errMsg);
  }

  int c_writeColumnToParquet(const char* filename, void* chpl_arr,
                             int64_t colnum, const char* dsetname, int64_t numelems,
                             int64_t rowGroupSize, int64_t dtype, int64_t compression,
                             char** errMsg) {
    return cpp_writeColumnToParquet(filename, chpl_arr, colnum, dsetname,
                                    numelems, rowGroupSize, dtype, compression,
                                    errMsg);
  }
  
  int c_writeStrColumnToParquet(const char* filename, void* chpl_arr, void* chpl_offsets,
                                const char* dsetname, int64_t numelems,
                                int64_t rowGroupSize, int64_t dtype, int64_t compression,
                                char** errMsg) {
    return cpp_writeStrColumnToParquet(filename, chpl_arr, chpl_offsets,
                                       dsetname, numelems, rowGroupSize, dtype, compression, errMsg);
  }

  int c_writeListColumnToParquet(const char* filename, void* chpl_segs, void* chpl_arr,
                                const char* dsetname, int64_t numelems,
                                int64_t rowGroupSize, int64_t dtype, int64_t compression,
                                char** errMsg) {
    return cpp_writeListColumnToParquet(filename, chpl_segs, chpl_arr,
                                       dsetname, numelems, rowGroupSize, dtype, compression, errMsg);
  }

  int c_writeStrListColumnToParquet(const char* filename, void* chpl_segs, void* chpl_offsets, void* chpl_arr,
                                const char* dsetname, int64_t numelems,
                                int64_t rowGroupSize, int64_t dtype, int64_t compression,
                                char** errMsg) {
    return cpp_writeStrListColumnToParquet(filename, chpl_segs, chpl_offsets, chpl_arr,
                                       dsetname, numelems, rowGroupSize, dtype, compression, errMsg);
  }

  int c_createEmptyParquetFile(const char* filename, const char* dsetname, int64_t dtype,
                               int64_t compression, char** errMsg) {
    return cpp_createEmptyParquetFile(filename, dsetname, dtype, compression, errMsg);
  }

  int c_createEmptyListParquetFile(const char* filename, const char* dsetname, int64_t dtype,
                               int64_t compression, char** errMsg) {
    return cpp_createEmptyListParquetFile(filename, dsetname, dtype, compression, errMsg);
  }

  int c_appendColumnToParquet(const char* filename, void* chpl_arr,
                              const char* dsetname, int64_t numelems,
                              int64_t dtype, int64_t compression,
                              char** errMsg) {
    return cpp_appendColumnToParquet(filename, chpl_arr,
                                     dsetname, numelems,
                                     dtype, compression,
                                     errMsg);
  }

  int64_t c_getStringColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, char** errMsg) {
    return cpp_getStringColumnNumBytes(filename, colname, chpl_offsets, numElems, startIdx, errMsg);
  }

  int64_t c_getListColumnSize(const char* filename, const char* colname, void* chpl_seg_sizes, int64_t numElems, int64_t startIdx, char** errMsg) {
    return cpp_getListColumnSize(filename, colname, chpl_seg_sizes, numElems, startIdx, errMsg);
  }

  int64_t c_getStringColumnNullIndices(const char* filename, const char* colname, void* chpl_nulls, char** errMsg) {
    return cpp_getStringColumnNullIndices(filename, colname, chpl_nulls, errMsg);
  }

  const char* c_getVersionInfo(void) {
    return cpp_getVersionInfo();
  }

  int c_getDatasetNames(const char* filename, char** dsetResult, bool readNested, char** errMsg) {
    return cpp_getDatasetNames(filename, dsetResult, readNested, errMsg);
  }

  void c_free_string(void* ptr) {
    cpp_free_string(ptr);
  }

  int c_writeMultiColToParquet(const char* filename, void* column_names, 
                                void** ptr_arr, void** offset_arr, void* objTypes, void* datatypes,
                                void* segArr_sizes, int64_t colnum, int64_t numelems, int64_t rowGroupSize,
                                int64_t compression, char** errMsg){
    return cpp_writeMultiColToParquet(filename, column_names, ptr_arr, offset_arr, objTypes, datatypes, segArr_sizes, colnum, numelems, rowGroupSize, compression, errMsg);
  }
}
