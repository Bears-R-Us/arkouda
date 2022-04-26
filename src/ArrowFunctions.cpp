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
    else if(myType->id() == arrow::Type::INT32)
      return ARROWINT32;
    else if(myType->id() == arrow::Type::UINT64)
      return ARROWUINT64;
    else if(myType->id() == arrow::Type::UINT32)
      return ARROWUINT32;
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

int64_t cpp_getStringColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, char** errMsg) {
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

        int64_t numRead = 0;
        while (ba_reader->HasNext() && numRead < numElems) {
          parquet::ByteArray value;
          (void)ba_reader->ReadBatch(1, &definition_level, nullptr, &value, &values_read);
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
      return byteSize;
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
        reader->Skip(startIdx);

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
        reader->Skip(startIdx);

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
        reader->Skip(startIdx);

        while (reader->HasNext() && i < numElems) {
          if((numElems - i) < batchSize)
            batchSize = numElems - i;
          (void)reader->ReadBatch(batchSize, nullptr, nullptr, &chpl_ptr[i], &values_read);
          i+=values_read;
        }
      } else if(ty == ARROWSTRING) {
        auto chpl_ptr = (unsigned char*)chpl_arr;
        int16_t definition_level;
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
        reader->Skip(startIdx);

        float* tmpArr = (float*)malloc(batchSize * sizeof(float));
        while (reader->HasNext() && i < numElems) {
          if((numElems - i) < batchSize)
            batchSize = numElems - i;
          // Can't read directly into chpl_ptr because it is a double
          (void)reader->ReadBatch(batchSize, nullptr, nullptr, tmpArr, &values_read);
          for (int64_t j = 0; j < values_read; j++)
            chpl_ptr[i+j] = (double)tmpArr[j];
          i+=values_read;
        }
        free(tmpArr);
      } else if(ty == ARROWDOUBLE) {
        auto chpl_ptr = (double*)chpl_arr;
        parquet::DoubleReader* reader =
          static_cast<parquet::DoubleReader*>(column_reader.get());
        reader->Skip(startIdx);

        while (reader->HasNext() && i < numElems) {
          if((numElems - i) < batchSize)
            batchSize = numElems - i;
          (void)reader->ReadBatch(batchSize, nullptr, nullptr, &chpl_ptr[i], &values_read);
          i+=values_read;
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

int cpp_writeColumnToParquet(const char* filename, void* chpl_arr,
                             int64_t colnum, const char* dsetname, int64_t numelems,
                             int64_t rowGroupSize, int64_t dtype, bool compressed,
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
    if(compressed) {
      builder.compression(parquet::Compression::SNAPPY);
      builder.encoding(parquet::Encoding::RLE);
    }
    std::shared_ptr<parquet::WriterProperties> props = builder.build();

    std::shared_ptr<parquet::ParquetFileWriter> file_writer =
      parquet::ParquetFileWriter::Open(out_file, schema, props);

    int64_t i = 0;
    int64_t numLeft = numelems;

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
                                int64_t rowGroupSize, int64_t dtype, bool compressed,
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
    if(compressed) {
      builder.compression(parquet::Compression::SNAPPY);
      builder.encoding(parquet::Encoding::RLE);
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

int cpp_createEmptyParquetFile(const char* filename, const char* dsetname, int64_t dtype,
                               bool compressed, char** errMsg) {
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
    if(compressed) {
      builder.compression(parquet::Compression::SNAPPY);
      builder.encoding(parquet::Encoding::RLE);
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
                              int64_t dtype, bool compressed,
                              char** errMsg) {
  try {
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
        ARROWSTATUS_OK(builder.Append(tmp_str));
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
    arrow::ArrayVector arrays;
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

int cpp_getDatasetNames(const char* filename, char** dsetResult, char** errMsg) {
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
         sc->field(i)->type()->id() == arrow::Type::UINT64 ||
         sc->field(i)->type()->id() == arrow::Type::UINT32 ||
         sc->field(i)->type()->id() == arrow::Type::TIMESTAMP ||
         sc->field(i)->type()->id() == arrow::Type::BOOL ||
         sc->field(i)->type()->id() == arrow::Type::STRING ||
         sc->field(i)->type()->id() == arrow::Type::BINARY ||
         sc->field(i)->type()->id() == arrow::Type::FLOAT ||
         sc->field(i)->type()->id() == arrow::Type::DOUBLE) {
        if(!first)
          fields += ("," + sc->field(i)->name());
        else
          fields += (sc->field(i)->name());
        first = false;
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

  int c_readColumnByName(const char* filename, void* chpl_arr, const char* colname, int64_t numElems, int64_t startIdx, int64_t batchSize, char** errMsg) {
    return cpp_readColumnByName(filename, chpl_arr, colname, numElems, startIdx, batchSize, errMsg);
  }

  int c_getType(const char* filename, const char* colname, char** errMsg) {
    return cpp_getType(filename, colname, errMsg);
  }

  int c_writeColumnToParquet(const char* filename, void* chpl_arr,
                             int64_t colnum, const char* dsetname, int64_t numelems,
                             int64_t rowGroupSize, int64_t dtype, bool compressed,
                             char** errMsg) {
    return cpp_writeColumnToParquet(filename, chpl_arr, colnum, dsetname,
                                    numelems, rowGroupSize, dtype, compressed,
                                    errMsg);
  }
  
  int c_writeStrColumnToParquet(const char* filename, void* chpl_arr, void* chpl_offsets,
                                const char* dsetname, int64_t numelems,
                                int64_t rowGroupSize, int64_t dtype, bool compressed,
                                char** errMsg) {
    return cpp_writeStrColumnToParquet(filename, chpl_arr, chpl_offsets,
                                       dsetname, numelems, rowGroupSize, dtype, compressed, errMsg);
  }

  int c_createEmptyParquetFile(const char* filename, const char* dsetname, int64_t dtype,
                               bool compressed, char** errMsg) {
    return cpp_createEmptyParquetFile(filename, dsetname, dtype, compressed, errMsg);
  }

  int c_appendColumnToParquet(const char* filename, void* chpl_arr,
                              const char* dsetname, int64_t numelems,
                              int64_t dtype, bool compressed,
                              char** errMsg) {
    return cpp_appendColumnToParquet(filename, chpl_arr,
                                     dsetname, numelems,
                                     dtype, compressed,
                                     errMsg);
  }

  int64_t c_getStringColumnNumBytes(const char* filename, const char* colname, void* chpl_offsets, int64_t numElems, int64_t startIdx, char** errMsg) {
    return cpp_getStringColumnNumBytes(filename, colname, chpl_offsets, numElems, startIdx, errMsg);
  }

  int64_t c_getStringColumnNullIndices(const char* filename, const char* colname, void* chpl_nulls, char** errMsg) {
    return cpp_getStringColumnNullIndices(filename, colname, chpl_nulls, errMsg);
  }

  const char* c_getVersionInfo(void) {
    return cpp_getVersionInfo();
  }

  int c_getDatasetNames(const char* filename, char** dsetResult, char** errMsg) {
    return cpp_getDatasetNames(filename, dsetResult, errMsg);
  }

  void c_free_string(void* ptr) {
    cpp_free_string(ptr);
  }
}
