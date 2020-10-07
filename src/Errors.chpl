

module Errors {

    /*
     * Generates an error message that provides a fuller context to the error
     * by including the line number, proc name, and module name from which the 
     * Error was thrown.
     */
    class ErrorWithContext: Error {   
        proc init(msg : string, lineNumber: int, routineName: string, 
                                                         moduleName: string) {
            try! super.init("Line %t In %s.%s: %s".format(lineNumber,
                                                          moduleName,
                                                          routineName,
                                                          msg));
        }
        
        proc init() {
            super.init();
        }
    }
    
    /*
     * The DatasetNotFoundError is thrown if there is no dataset in the file
     * being accessed.
     */
    class DatasetNotFoundError: ErrorWithContext { 
        proc init(msg : string, lineNumber: int, routineName: string, 
                                                           moduleName: string) { 
           super.init(msg,lineNumber,routineName,moduleName); 
        } 

        proc init(){ super.init(); }
    }
    
    proc getErrorWithContext(lineNumber: int, moduleName: string, 
                                      routineName: string, msg: string, errorClass: string) throws {
        select errorClass {
            when "DatasetNotFoundError"          { return new owned 
                                                          DatasetNotFoundError(msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName); }
                                                          
            otherwise { throw new owned Error(); }
        }
    }

}