module ServerErrors {

    private use IO;

    class OutOfBoundsError: Error {}

    /*
     * Generates an error message that provides a fuller context to the error
     * by including the line number, proc name, and module name from which the 
     * Error was thrown.
     */
    class ErrorWithContext: Error {  
        var lineNumber: int;
        var routineName: string;
        var moduleName: string;
        var errorClass: string;
        var publishMsg: string;
        
        /*
         * Accepts parameters that are used to generate the detailed, context-rich
         * error message accessible via ErrorWithContext.message() as well as the 
         * client-formatted error message accessible via ErrorWithContext.publish()
         */
        proc init(msg : string, lineNumber: int, routineName: string, 
                      moduleName: string, errorClass: string='ErrorWithContext') {
            super.init(generateErrorContext(msg=msg, lineNumber=lineNumber,
              moduleName=moduleName, routineName=routineName, errorClass=errorClass));

            this.lineNumber = lineNumber;
            this.routineName = routineName;
            this.moduleName = moduleName;
            this.errorClass = errorClass;
            this.publishMsg = msg;
        }
        
        proc init() {
            super.init();
        }
        
        /*
         * Returns only the msg init parameter element prepended with "Error: ",
         * which can be used to report errors back to the Arkouda client in a format
         * understandable to front-end developers as well as users.
         */
        proc publish() : string {
            return "Error: " + publishMsg;
        }
    }
    
    /*
     * The DatasetNotFoundError is thrown if there is no dataset in the file
     * being accessed.
     */
    class DatasetNotFoundError: ErrorWithContext { 

        proc init(msg : string, lineNumber: int, routineName: string, 
                                                           moduleName: string) { 
           super.init(msg,lineNumber,routineName,moduleName,errorClass='DatasetNotFoundError'); 
        } 

        proc init(){ super.init(); }
    }
    
    /*
     * The WriteModeError is thrown if a file save operation is executed in append mode
     * on a brand new file lacking any current datasets.
     */
    class WriteModeError: ErrorWithContext { 

        proc init(msg : string, lineNumber: int, routineName: string, 
                                                           moduleName: string) { 
           super.init(msg,lineNumber,routineName,moduleName,errorClass='WriteModeError'); 
        } 

        proc init(){ super.init(); }
    }
 
    /*
     * The NotHDF5FileError is thrown if it is determined a file is not HDFF file.
     */
    class NotHDF5FileError: ErrorWithContext { 

        proc init(msg : string, lineNumber: int, routineName: string, 
                                                           moduleName: string) { 
           super.init(msg,lineNumber,routineName,moduleName,errorClass='NotHDF5FileError'); 
        } 

        proc init(){ super.init(); }
    }

    /*
     * The HDF5FileFormatError is thrown if there is an error in parsing the HDF5 file.
     */
    class HDF5FileFormatError: ErrorWithContext { 

        proc init(msg : string, lineNumber: int, routineName: string, 
                                                           moduleName: string) { 
           super.init(msg,lineNumber,routineName,moduleName,errorClass='HDF5FileFormatError'); 
        } 

        proc init(){ super.init(); }
    }

    /*
     * The MismatchedAppendError is thrown if an attempt is made to append a dataset to
     * an HDF5 file where the number of locales for the current Arkouda instance differs
     * from the number of locales in the Arkouda instance that wrote the original files.
     */
    class MismatchedAppendError: ErrorWithContext { 

        proc init(msg : string, lineNumber: int, routineName: string, 
                                                           moduleName: string) { 
           super.init(msg,lineNumber,routineName,moduleName,errorClass='MismatchedAppendError'); 
        } 

        proc init(){ super.init(); }
    }

    /*
     * The SegStringError is thrown if the file corresponding to the SegString lacks either the
     * SEGSTRING_OFFSET_NAME or SEGSTRING_VALUE_NAME dataset.
     */
    class SegStringError: ErrorWithContext { 

        proc init(msg : string, lineNumber: int, routineName: string, 
                                                           moduleName: string) { 
           super.init(msg,lineNumber,routineName,moduleName,errorClass='SegStringError'); 
        } 

        proc init(){ super.init(); }
    }
    
    /*
     * The ArgumentError is thrown if there is a problem with 1.n arguments passed
     * into a function.
     */
    class ArgumentError: ErrorWithContext { 

        proc init(msg : string, lineNumber: int, routineName: string, 
                                                           moduleName: string) { 
           super.init(msg,lineNumber,routineName,moduleName,errorClass='ArgumentError'); 
        } 

        proc init(){ super.init(); }
    }

    /*
     * The NotImplementedError is thrown if the requested operation has not been implemented
     * for the specified data type(s) and/or command type.
     */
    class NotImplementedError: ErrorWithContext { 

        proc init(msg : string, lineNumber: int, routineName: string, 
                                                           moduleName: string) { 
           super.init(msg,lineNumber,routineName,moduleName,errorClass='NotImplementedError'); 
        } 

        proc init(){ super.init(); }
    }

    /*
     * The UnknownSymbolError is thrown if there is not entry in the SymTab.
     */
    class UnknownSymbolError: ErrorWithContext { 

        proc init(msg : string, lineNumber: int, routineName: string, 
                                                           moduleName: string) { 
           super.init(msg,lineNumber,routineName,moduleName,errorClass='UnknownSymbolError'); 
        } 

        proc init(){ super.init(); }
    }

    /*
     * The UnsupportedOSError is thrown if a function cannot be executed on the host OS.
     */
    class UnsupportedOSError: ErrorWithContext {

        proc init(msg : string, lineNumber: int, routineName: string,
                                                           moduleName: string) {
           super.init(msg,lineNumber,routineName,moduleName,errorClass='UnsupportedOSError');
        }

        proc init(){ super.init(); }
    }

    /*
     * The IOError is thrown if there is an error in IO code.
     */
    class IOError: ErrorWithContext {

        proc init(msg : string, lineNumber: int, routineName: string,
                                                           moduleName: string) {
           super.init(msg,lineNumber,routineName,moduleName,errorClass='IOError');
        }

        proc init(){ super.init(); }
    }

    /*
     * The OverMemoryLimitError is thrown if the projected memory required for a method
     * invocation will exceed available, free memory on 1..n locales
     */
    class OverMemoryLimitError: ErrorWithContext {

        proc init(msg : string, lineNumber: int, routineName: string,
                                                           moduleName: string) {
           super.init(msg,lineNumber,routineName,moduleName,errorClass='OverMemoryLimitError');
        }

        proc init(){ super.init(); }
    }

     /*
     * The ConfigurationError if the current instance of the server was not
     * configured to complete a requested operation.
     */
    class ConfigurationError: ErrorWithContext {

        proc init(msg : string, lineNumber: int, routineName: string,
                                                           moduleName: string) {
           super.init(msg,lineNumber,routineName,moduleName,errorClass='ConfigurationError');
        }

        proc init(){ super.init(); }
    }

    /*
     * Generates a detailed, context-rich error message for errors such as instances of 
     * built-in Chapel Errors in a format that matches the Arkouda ErrorWithContext
     * error message format. 
     */
    proc generateErrorContext(msg: string, lineNumber: int, moduleName: string, routineName: string, 
                                        errorClass: string="ErrorWithContext") : string {
        return "".join(errorClass, " Line ", lineNumber:string, " In ",
                       moduleName, ".", routineName, ": ", msg);
    }
 
    /*
     * Factory method for generating ErrorWithContext objects that include an error
     * message as well as the line number, routine name, and module name where the 
     * error was thrown.
     */
    proc getErrorWithContext(lineNumber: int, moduleName: string, 
                    routineName: string, msg: string, errorClass: string) throws {
        select errorClass {
            when "ErrorWithContext"             { return new owned 
                                                          ErrorWithContext(msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName); }
            when "DatasetNotFoundError"          { return new owned 
                                                          DatasetNotFoundError(msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName); }
            when "NotHDF5FileError"              { return new owned 
                                                          NotHDF5FileError(msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName); }
            when "HDF5FileFormatError"           { return new owned 
                                                          HDF5FileFormatError(msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName); }
            when "WriteModeError"                { return new owned 
                                                          WriteModeError(msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName); }
            when "MismatchedAppendError"         { return new owned 
                                                          MismatchedAppendError(msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName); }
            when "ArgumentError"                 { return new owned 
                                                          ArgumentError(msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName); }
            when "FileNotFoundError"             { return new owned 
                                                          FileNotFoundError(generateErrorContext(
                                                          msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName, 
                                                          errorClass=errorClass)); }       
            when "PermissionError"               { return new owned 
                                                          PermissionError(generateErrorContext(
                                                          msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName, 
                                                          errorClass=errorClass)); }
            when "IllegalArgumentError"          { return new owned 
                                                          IllegalArgumentError(generateErrorContext(
                                                          msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName, 
                                                          errorClass=errorClass)); }
            when "UnknownSymbolError"            { return new owned 
                                                          UnknownSymbolError(msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName); }
            when "IOError"                       { return new owned
                                                          IOError(msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName); }

            when "OverMemoryLimitError"          { return new owned
                                                          OverMemoryLimitError(msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName); 
                                                 }

            otherwise                            { return new owned 
                                                          Error(generateErrorContext(
                                                          msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName, 
                                                          errorClass=errorClass)); }
        }
    }
}
