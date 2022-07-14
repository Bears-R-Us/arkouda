use UnitTest;
use TestBase;
use IO;
use FileIO;
use GenSymIO;
use HDF5Msg;
use Message;
use SegmentedString;
import FileSystem;

const nb_str:string = b"\x00".decode(); // create null_byte string
const nb_byt:bytes = b"\x00"; // create null_byte
const s = nb_str.join("one", "two", "three", "four", "five") + nb_str;
const s_values: [0..#s.numBytes] uint(8) = for b in s.chpl_bytes() do b;
const s_offsets = [0, 4, 8, 14, 19];
const tmp_file_path = "resources/tmp_UnitTestHDF5";


/**
 * Utility method to clean up files
 */
proc removeFile(fn:string) {
    try {
        if (FileSystem.exists(fn)) {
            unlink(fn);
        }
    } catch {}
}

/**
 * Utility method to compare the contents of two arrays
 */
proc compareArrays(test: borrowed Test, expected, actual, msg:string="", debug:bool = false) {
    for (ex, ac) in zip(expected, actual) {
        if(debug) {
            writeln("msg:", msg, " expected:", ex, " - actual:", ac);
        }
        test.assertTrue(ex == ac);
    }
}

/**
 * Unit test to test proper calculation of offsets/segments from a SegStrings values array.
 */
proc test_segmentedCalcOffsets(t: borrowed Test) {
    var calculated = segmentedCalcOffsets(s_values, s_values.domain);
    compareArrays(t, s_offsets, calculated);
}

/**
 * Unit test for `tohdfMsg`
 * Tests a SegString being written to HDF5, toggling the option to write the offsets array
 */
proc test_tohdfMsg(t: borrowed Test) throws {
    try {
        var fWithOffsets = tmp_file_path + "_withOffsets";
        var fNoOffsets = tmp_file_path + "_noOffsets";
        defer { // Make sure we clean up regardless of outcome
            removeFile(fWithOffsets + "_LOCALE0000");
            removeFile(fNoOffsets + "_LOCALE0000");
        }
        var offsets = segmentedCalcOffsets(s_values, s_values.domain);
        var st = new owned SymTab();
        var segString = getSegString(s_offsets, s_values, st); // from SegmentedString
        var cmd = "tohdf";
        // payload -> arrayName, dsetName, modeStr, jsonfile, dataType, segsName, writeOffsetsFlag
        var payload = segString.name + " strings 0 [" + Q+fWithOffsets+Q + "] strings " + segString.name + " true";
        var msg = tohdfMsg(cmd, payload, st);
        t.assertTrue(msg.msg.strip() == "wrote array to file");
        
        payload = segString.name + " strings 0 [" + Q+fNoOffsets+Q + "] strings " + segString.name + " false";
        msg = tohdfMsg(cmd, payload, st);
        t.assertTrue(msg.msg.strip() == "wrote array to file");
    }

}

/**
 * Unit test for `readAllHdfMsg`
 * Tests reading a SegString from disk using included test files.
 * One file has offsets/segments array included and the other does not so we can test
 * reading them from disk and calculating from null-byte terminators in the values.
 */
proc test_readAllHdfMsg(t: borrowed Test) throws {
    var st = new owned SymTab();
    var readNoOffsets = "resources/UnitTestHDF5_noOffsets_LOCALE0000";
    var readWithOffsets = "resources/UnitTestHDF5_withOffsets_LOCALE0000";
    // f"{strictTypes} {len(datasets)} {len(filenames)} {allow_errors} {calc_string_offsets} {json.dumps(datasets)} | {json.dumps(filenames)}"
    // Exercise the basic read functionality with & without offsets and then again using dataset names
    var payload = "false 1 1 false false [\"strings\"] | [\"resources/UnitTestHDF5_withOffsets_LOCALE0000\"]";
    var msg = readAllHdfMsg("readAllHdfMsg", payload, st);
    t.assertTrue(msg.msg.find("str 5 1 (5,) 1+created bytes.size 24") > 0);

    // Same test built with calcOffsets true & noOffsets file
    payload = "false 1 1 false true [\"strings\"] | [\"resources/UnitTestHDF5_noOffsets_LOCALE0000\"]";
    msg = readAllHdfMsg("readAllHdfMsg", payload, st);
    t.assertTrue(msg.msg.find("str 5 1 (5,) 1+created bytes.size 24") > 0);
}

/**
 * To include tests they must be called from the main procedure
 */
proc main() {
    var t = new Test();
    test_segmentedCalcOffsets(t);
    test_tohdfMsg(t);
    test_readAllHdfMsg(t);
}
