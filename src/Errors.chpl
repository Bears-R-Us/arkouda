module Errors {

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
         
        proc init(msg : string, lineNumber: int, routineName: string, 
                      moduleName: string, errorClass: string='ErrorWithContext') {
            try! super.init("%s Line %t In %s.%s: %s".format(errorClass,lineNumber,
                                                          moduleName,
                                                          routineName,
                                                          msg));
            this.lineNumber = lineNumber;
            this.routineName = routineName;
            this.moduleName = moduleName;
            this.errorClass = errorClass;
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
           super.init(msg,lineNumber,routineName,moduleName,errorClass='DatasetNotFoundError'); 
        } 

        proc init(){ super.init(); }
    }
 
    /*
     * This function is used to generate a detailed, context-rich error message for errors such as
     * instances of built-in Chapel Errors in a format that matches the Arkouda ErrorWithContext
     * error message format. 
     */
    proc generateErrorContext(msg: string, lineNumber: int, moduleName: string, routineName: string, 
                                        errorClass: string="ErrorWithContext") : string {
        return try! "%s %t %s:%s %s".format(errorClass,lineNumber,moduleName,routineName,msg);
    }
 
    
    proc getErrorWithContext(lineNumber: int, moduleName: string, 
                                      routineName: string, msg: string, errorClass: string): Error throws {
        select errorClass {
            when "DatasetNotFoundError"          { return new owned 
                                                          DatasetNotFoundError(msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName); }
            when "FileNotFoundError"             { return new owned 
                                                          FileNotFoundError(generateErrorContext(
                                                          msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName, 
                                                          errorClass='FileNotFoundError')); }       
            when "PermissionError"               { return new owned 
                                                          PermissionError(generateErrorContext(
                                                          msg=msg,
                                                          lineNumber=lineNumber,
                                                          routineName=routineName,
                                                          moduleName=moduleName, 
                                                          errorClass='PermissionError')); }                               
            otherwise { throw new owned Error(); }
        }
    }
}