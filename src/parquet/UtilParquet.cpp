#include "UtilParquet.h"

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

static std::map<int, std::shared_ptr<parquet::ParquetFileReader>> globalFiles;
static std::map<int, std::shared_ptr<parquet::RowGroupReader>> globalRowGroupReaders;
static std::map<int, std::shared_ptr<parquet::ColumnReader>> globalColumnReaders;

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

int cpp_getPrecision(const char* filename, const char* colname, char** errMsg) {
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

    const auto& decimal_type = static_cast<const ::arrow::DecimalType&>(*sc->field(idx)->type());
    const int64_t precision = decimal_type.precision();

    return precision;
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
            myType->id() == arrow::Type::BINARY ||
            myType->id() == arrow::Type::LARGE_STRING)
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
                f_type->id() == arrow::Type::BINARY ||
                f_type->id() == arrow::Type::LARGE_STRING)  // Verify that this is functional as expected
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
        fields.push_back(parquet::schema::PrimitiveNode::Make(cname_ptr[i], parquet::Repetition::OPTIONAL, parquet::Type::BYTE_ARRAY, parquet::ConvertedType::NONE));
      }
    }
  }
  return std::static_pointer_cast<parquet::schema::GroupNode>(
      parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));
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
         sc->field(i)->type()->id() == arrow::Type::DECIMAL ||
         sc->field(i)->type()->id() == arrow::Type::LARGE_STRING
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

void cpp_openFile(const char* filename, int64_t idx) {
  std::shared_ptr<parquet::ParquetFileReader> parquet_reader =
    parquet::ParquetFileReader::OpenFile(filename, false);
  globalFiles[idx] = parquet_reader;
}

void cpp_createRowGroupReader(int64_t rowGroup, int64_t readerIdx) {
  std::shared_ptr<parquet::RowGroupReader> row_group_reader =
    globalFiles[readerIdx]->RowGroup(rowGroup);
  globalRowGroupReaders[readerIdx] = row_group_reader;
}

void cpp_createColumnReader(const char* colname, int64_t readerIdx) {
  
  std::shared_ptr<parquet::FileMetaData> file_metadata = globalFiles[readerIdx]->metadata();
  auto idx = file_metadata -> schema() -> ColumnIndex(colname);

  std::shared_ptr<parquet::ColumnReader> column_reader;
  column_reader = globalRowGroupReaders[readerIdx]->Column(idx);
  globalColumnReaders[readerIdx] = column_reader;
}



int cpp_getNumRowGroups(int64_t readerIdx) {
  std::shared_ptr<parquet::FileMetaData> file_metadata = globalFiles[readerIdx]->metadata();
  return file_metadata->num_row_groups();
}

void cpp_freeMapValues(void* row) {
  parquet::ByteArray* string_values =
    static_cast<parquet::ByteArray*>(row);
  free(string_values);
  globalColumnReaders.clear();
  globalRowGroupReaders.clear();
  globalFiles.clear();
}

int cpp_readParquetColumnChunks(const char* filename, int64_t batchSize, int64_t numElems,
                                int64_t readerIdx, int64_t* numRead,
                                void** outData, bool* containsNulls, char** errMsg) {
  try {
    auto reader = static_cast<parquet::ByteArrayReader*>(globalColumnReaders[readerIdx].get());
    parquet::ByteArray* string_values =
      (parquet::ByteArray*)malloc(numElems*sizeof(parquet::ByteArray));
    std::vector<int16_t> definition_level(batchSize);
    int64_t values_read = 0;
    int64_t total_read = 0;
    while(reader->HasNext() && total_read < numElems) {
      if((numElems - total_read) < batchSize)
        batchSize = numElems - total_read;
      // adding 1 to definition level, since the first value indicates if null values
      (void)reader->ReadBatch(batchSize, definition_level.data(), nullptr, string_values + total_read, &values_read);
      for(int i = 0; i < values_read; i++) {
        if(definition_level[i] == 0)
          *containsNulls = true;
      }
      total_read += values_read;
    }
    *numRead = total_read;
    *outData = (void*)string_values;
    return 0;
  } catch (const std::exception& e) {
    *errMsg = strdup(e.what());
    return ARROWERROR;
  }
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

  int c_getType(const char* filename, const char* colname, char** errMsg) {
    return cpp_getType(filename, colname, errMsg);
  }

  int c_getListType(const char* filename, const char* colname, char** errMsg) {
    return cpp_getListType(filename, colname, errMsg);
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

  int c_getPrecision(const char* filename, const char* colname, char** errMsg) {
    return cpp_getPrecision(filename, colname, errMsg);
  }

  void c_openFile(const char* filename, int64_t idx) {
    cpp_openFile(filename, idx);
  }

  void c_createRowGroupReader(int64_t rowGroup, int64_t readerIdx) {
    return cpp_createRowGroupReader(rowGroup, readerIdx);
  }

  void c_createColumnReader(const char* colname, int64_t readerIdx) {
    cpp_createColumnReader(colname, readerIdx);
  }

  int c_getNumRowGroups(int64_t readerIdx) {
    return cpp_getNumRowGroups(readerIdx);
  }

  void c_freeMapValues(void* row) {
    cpp_freeMapValues(row);
  }

  int c_readParquetColumnChunks(const char* filename, int64_t batchSize, int64_t numElems,
                                int64_t readerIdx, int64_t* numRead,
                                void** outData, bool* containsNulls, char** errMsg) {
    return cpp_readParquetColumnChunks(filename, batchSize, numElems, readerIdx,
                                       numRead, outData, containsNulls, errMsg);
  }
}
