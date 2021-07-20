module io {
    use IO only file, opentmp;

    config const SEGARRAY_OFFSET_NAME = "segments";
    config const SEGARRAY_VALUE_NAME = "values";

    /*
     * Ensure the file is closed, disregard errors
     */
    proc ensureClose(tmpf:file): bool {
        var success = true;
        try {
            tmpf.close();
        } catch {
            success = false;
        }
        return success;
    }

    //----------------------------------------
    // Module Includes
    include module FileIO;
    include module GenSymIO;
    include module GenHdf5;

}