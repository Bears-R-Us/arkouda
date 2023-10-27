#include "read-parquet.h"

int getNumRows(std::string filename) {
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
    parquet::ParquetFileReader::OpenFile(filename, false);

  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
  const int numElems = file_metadata -> num_rows();
  return numElems;
}

int readColumnByName(std::string filename, std::string colname, int64_t* arr, const int numElems, int64_t batchSize) {
  try {
    std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
      parquet::ParquetFileReader::OpenFile(filename, false);

    std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();
    const int num_row_groups = file_metadata->num_row_groups();

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

void readColumns(std::string filename, std::string colname, int num_cols, int64_t* arr, const int numElems, int64_t batchSize) {
  for(int i = 1; i <= num_cols; i++) {
    std::string col = colname;
    col.append(std::to_string(i));
    readColumnByName(filename, col, arr, numElems, batchSize);
  }
}

int main(int argc, char** argv) {
  std::string filename = argv[1];
  std::string colname = argv[2];
  int batchSize = atoi(argv[3]);
  int num_cols = atoi(argv[4]);

  const int numElems = getNumRows(filename);
  int64_t* arr = (int64_t*)malloc(numElems*sizeof(int64_t));

  std::cout << "Reading " << num_cols << " columns using low-level API: ";
  auto start = std::chrono::high_resolution_clock::now();
  readColumns(filename, "col", num_cols, arr, numElems, batchSize);
  auto finish = std::chrono::high_resolution_clock::now();
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start);
  std::cout << milliseconds.count()/1000.0 << "s\n";
  
  return 0;
}
