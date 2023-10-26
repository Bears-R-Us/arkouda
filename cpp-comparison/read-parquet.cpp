#include "asd.h"

int readColumnByName(const char* filename, const char* colname, int64_t* arr, int64_t numElems, int64_t batchSize) {
  try {
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
      
      column_reader = row_group_reader->Column(idx);

      parquet::Int64Reader* reader =
        static_cast<parquet::Int64Reader*>(column_reader.get());

      while (reader->HasNext() && i < numElems) {
        if((numElems - i) < batchSize)
          batchSize = numElems - i;
        (void)reader->ReadBatch(batchSize, nullptr, nullptr, &arr[i], &values_read);
        i+=values_read;
      }
    }
    return 0;
  } catch(const std::exception& e) {
    return -1;
  }
}

int readColumnByName2(const char* filename, const char* colname, int64_t* arr, int64_t numElems, int64_t batchSize) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(infile,
                          arrow::io::ReadableFile::Open(filename,
                                                        arrow::default_memory_pool()));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
  std::shared_ptr<arrow::ChunkedArray> array;
  PARQUET_THROW_NOT_OK(reader->ReadColumn(0, &array));
  return 0;
}

int main(int argc, char** argv) {
  const char* filename = argv[1];
  const char* colname = argv[2];
  int batchSize = atoi(argv[3]);

  int64_t numElems = 1000000000;
  int64_t* arr = (int64_t*)malloc(numElems*sizeof(int64_t));

  std::cout << "Reading using low-level API: ";
  auto start = std::chrono::high_resolution_clock::now();
  readColumnByName(filename, colname, arr, numElems, batchSize);
  auto finish = std::chrono::high_resolution_clock::now();

  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start);
  std::cout << milliseconds.count()/1000.0 << "s\n";

  std::cout << "Reading using standard interface: ";
  start = std::chrono::high_resolution_clock::now();
  readColumnByName2(filename, colname, arr, numElems, batchSize);
  finish = std::chrono::high_resolution_clock::now();

  milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start);
  std::cout << milliseconds.count()/1000.0 << "s\n";
  
  return 0;
}
