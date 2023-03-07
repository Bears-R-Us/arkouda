use UnitTest;
use TestBase;
use IO;
use FileIO;
use GenSymIO;

proc getFirstEightBytesFromFile(path:string):bytes throws {
    //var f:file = open("resources/sample.parquet", ioMode.r);
    var f:file = open(path, ioMode.r);
    defer { f.close(); }
    var reader = f.reader(kind=ionative);
    var header:bytes;
    if (reader.binary()) {
        reader.readbytes(header, 8);
    } else {
        throw new Error("File reader was not in binary mode");
    }
    return header;
}

proc test_getFileTypeByMagic(test: borrowed Test) throws {
    var header:bytes;

    // Bounds check on zero byte header
    test.assertTrue(getFileTypeByMagic(header) == FileType.UNKNOWN);

    // Test a sample Arrow file, expect it to match correctly
    header = getFirstEightBytesFromFile("resources/sample.arrow");
    test.assertTrue(getFileTypeByMagic(header) == FileType.ARROW);     // True
    test.assertFalse(getFileTypeByMagic(header) == FileType.HDF5);     // False
    test.assertFalse(getFileTypeByMagic(header) == FileType.PARQUET);  // False
    
    // Test a sample Parquet file, expect it to match correctly
    header = getFirstEightBytesFromFile("resources/sample.parquet");
    test.assertTrue(getFileTypeByMagic(header) == FileType.PARQUET);   // True
    test.assertFalse(getFileTypeByMagic(header) == FileType.HDF5);     // False
    test.assertFalse(getFileTypeByMagic(header) == FileType.ARROW);    // False

    // Test a sample HDF5 file, expect it to match correctly
    header = getFirstEightBytesFromFile("resources/sample.hdf5");
    test.assertTrue(getFileTypeByMagic(header) == FileType.HDF5);      // True
    test.assertFalse(getFileTypeByMagic(header) == FileType.ARROW);    // False
    test.assertFalse(getFileTypeByMagic(header) == FileType.PARQUET);  // False

    // Test file magic does not match against a text file and matches UNKNOWN
    header = getFirstEightBytesFromFile("resources/sample.ascii.txt");
    test.assertTrue(getFileTypeByMagic(header) == FileType.UNKNOWN);   // True
    test.assertFalse(getFileTypeByMagic(header) == FileType.PARQUET);  // False
    test.assertFalse(getFileTypeByMagic(header) == FileType.ARROW);    // False
    test.assertFalse(getFileTypeByMagic(header) == FileType.HDF5);     // False

    // Note: We don't have an Apache Arrow file
}

proc main() {
    var t = new Test();
    test_getFileTypeByMagic(t);
}