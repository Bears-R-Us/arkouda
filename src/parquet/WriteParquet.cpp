#include "WriteParquet.h"
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

extern "C" {
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

  int c_writeMultiColToParquet(const char* filename, void* column_names, 
                               void** ptr_arr, void** offset_arr, void* objTypes, void* datatypes,
                               void* segArr_sizes, int64_t colnum, int64_t numelems, int64_t rowGroupSize,
                               int64_t compression, char** errMsg){
    return cpp_writeMultiColToParquet(filename, column_names, ptr_arr, offset_arr, objTypes, datatypes, segArr_sizes, colnum, numelems, rowGroupSize, compression, errMsg);
  }
}
