module MetricsMsg {
    use ServerConfig;
    use Reflection;
    use ServerErrors;
    use Logging;
    use List;
    use IO;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use Message;
    use MemDiagnostics;
    use NumPyDType;
    use Map;
    use Time;
    use IOUtils;

    enum MetricCategory{ALL,NUM_REQUESTS,RESPONSE_TIME,AVG_RESPONSE_TIME,TOTAL_RESPONSE_TIME,
                        TOTAL_MEMORY_USED,SYSTEM,SERVER,SERVER_INFO,NUM_ERRORS};
    enum MetricScope{GLOBAL,LOCALE,REQUEST,USER};
    enum MetricDataType{INT,REAL};

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const mLogger = new Logger(logLevel, logChannel);

    var metricScope = try! ServerConfig.getEnv(name='METRIC_SCOPE',default='MetricScope.REQUEST');
    
    var serverMetrics = new CounterTable();
    
    var requestMetrics = new CounterTable();
    
    var avgResponseTimeMetrics = new AverageMeasurementTable();
    
    var responseTimeMetrics = new MeasurementTable();
    
    var totalResponseTimeMetrics = new MeasurementTable();
    
    var totalMemoryUsedMetrics = new MeasurementTable();

    var users = new Users();
    
    var userMetrics = new UserMetrics();

    var errorMetrics = new CounterTable();

    record User {
        var name: string;
    }

    class Users {
        var users = new map(string,User);

        proc getUser(name: string) {
            if !this.users.contains(name) {
                var user = new User(name);
                users.add(name,user);
                return user;
            } else {
                return try! users[name];
            }
        }

        proc getUserNames() {
            return this.users.keys();
        }

        proc getUsers() {
            return this.users.values();
        }
    }
    
    class MetricValue {
        var realValue: real;
        var intValue: int(64);
        var dataType: MetricDataType;
        
        proc init(realValue : real) {
            this.realValue = realValue;
            this.dataType = MetricDataType.REAL;
        }
        
        proc init(intValue : int(64)) {
            this.intValue = intValue;
            this.dataType = MetricDataType.INT;
        }
        
        proc update(val) {
            if this.dataType == MetricDataType.INT {
                this.intValue += val;
            } else {
                this.realValue += val;
            }
        }
    }
    
    class AvgMetricValue : MetricValue {
        var numValues: int;
        var intTotal: int(64);
        var realTotal: real;
        
        proc update(val) {
            this.numValues += 1;

            if this.dataType == MetricDataType.INT {
                this.intTotal += val;
                this.realValue = this.intTotal/this.numValues;
            } else {
                this.realTotal += val;
                this.realValue = this.realTotal/this.numValues;
            }   
        }
    }

    class UserMetrics {
        var metrics = new map(keyType=User,valType=shared CounterTable);
        var users = new Users();

        proc getUserMetrics(user: User) : borrowed CounterTable {
            if this.metrics.contains(user: User) {
                return try! this.metrics[user];
            } else {
                var userMetrics = new shared CounterTable();
                this.metrics.add(user, userMetrics);
                return try! this.metrics[user];
            }
        }

        proc incrementPerUserRequestMetrics(userName: string, metricName: string, increment: int=1) {
            this.incrementNumRequestsPerCommand(userName,metricName,increment);
            this.incrementTotalNumRequests(userName,increment);
        }
        
        proc getPerUserNumRequestsPerCommandMetrics(userName: string) {
            var userMetrics = this.getUserMetrics(this.users.getUser(userName));
            var metrics = new list(owned UserMetric?);
            for (metric, value) in userMetrics.items() {
                metrics.pushBack(new UserMetric(name=metric,
                                              scope=MetricScope.USER,
                                              category=MetricCategory.NUM_REQUESTS,
                                              value=value,
                                              user=userName));
            }
            return metrics;
        }

        proc getPerUserNumRequestsPerCommandForAllUsersMetrics() {
            var metrics = new list(owned UserMetric?);
            for userName in this.users.getUserNames() {
                for metric in this.getPerUserNumRequestsPerCommandMetrics(userName) {
                    metrics.pushBack(metric);
                }
            }

            return metrics;
        }

        proc incrementNumRequestsPerCommand(userName: string, cmd: string, increment: int=1) {
            var userMetrics : borrowed CounterTable = this.getUserMetrics(this.users.getUser(userName));
            userMetrics.increment(cmd,increment);
        }   
        proc incrementTotalNumRequests(userName: string,increment: int=1) {
            var userMetrics : borrowed CounterTable = this.getUserMetrics(this.users.getUser(userName));
            userMetrics.increment('total',increment);
        }     
    } 

    /*
     * The MeasurementTable encapsulates real measurements
     */
    class MeasurementTable {
        var measurements = new map(string, real);

        /*
         * Returns the measurement corresponding to the metric name If the
         * metric does not exist, the metric value is set to 0.0 and is returned.
         */
        proc get(metric: string): real throws {
            var value: real;

            if !this.measurements.contains(metric) {
                value = 0.0;
                this.measurements.add(metric, value);
            } else {
                value = this.measurements(metric);
            }
            
            return value;
        }   

        /* 
         * Sets the metrics value
         */
        proc set(metric: string, measurement: real) throws {
            this.measurements.addOrReplace(metric, measurement);
        }

        /*
         * Returns the number of measurements in the MeasurementTable.s
         */
        proc size() {
            return this.measurements.size;
        }

        /* 
         * Adds a measurement to an existing measurement corresponding to the 
         * metric name, setting the value if the metric does not exist.
         */
        proc add(metric: string, measurement: real) throws {
            this.measurements.replace(metric,(this.get(metric) + measurement));
        }

        iter items() {
          for (key, val) in zip(measurements.keys(), measurements.values()) do
            yield(key, val);
        }
    }

    /* 
     * The AverageMeasurementTable extends the MeasurementTable by generating
     * values that are averages of incoming values for each metric.
     */
    class AverageMeasurementTable : MeasurementTable {
        //number of recorded measurements
        var numMeasurements = new map(string, int(64));
        
        // total value of measurements to be averaged for each metric measured.s
        var measurementTotals = new map(string, real);

        /*
         * Returns the number of measurements corresponding to a metric.
         */
        proc getNumMeasurements(metric: string) throws {
            if this.numMeasurements.contains(metric) {
                return this.numMeasurements(metric) + 1;
            } else {
                return 1;
            }
        }
        
        /*
         * Returns the sum of all measurements corresponding to a metric. Note:
         * this function is designed to invoked internally in order to 
         * calculate the avg measurement value corresponding to the metric.
         */
        proc getMeasurementTotal(metric: string) : real throws {
            var value: real;

            if !this.measurementTotals.contains(metric) {
                value = 0.0;
                this.measurementTotals.addOrReplace(metric, value);
            } else {
                value = this.measurementTotals(metric);
            }
            
            return value;
        }
        
        /*
         * The overridden add method updates the measurement value by doing the following:
         *
         * 1. adds the measurement to a running total measurement for the metric
         * 2. increments the number of measurements for the metric
         * 3. divides the updated total measurement by the number of measurements to
         *    to calculate the avg measurement
         */
        override proc add(metric: string, measurement: real) throws {
            var numMeasurements = getNumMeasurements(metric);
            var measurementTotal = getMeasurementTotal(metric);

            this.numMeasurements.addOrReplace(metric, numMeasurements);
            this.measurementTotals(metric) += measurement;

            var value: real = this.measurementTotals(metric)/numMeasurements;

            this.measurements.addOrReplace(metric, value);
        }
    }

    class CounterTable {
        var counts = new map(string, int);
       
        proc get(metric: string) : int {
            if !this.counts.contains(metric) {
                this.counts.add(metric,0);
                return 0;
            } else {
                return try! this.counts[metric];
            }
        }   
        
        proc set(metric: string, count: int) {
            this.counts.addOrReplace(metric,count);
        }
    
        proc increment(metric: string, increment: int=1) {
            var current = this.get(metric);
            
            // Set new metric value to current plus the increment
            this.set(metric, current+increment);
        }
    
        proc decrement(metric: string, increment: int=1) {
            var current = this.get(metric);
            
            /*
             * Set new metric value to current minus the increment 
             */
            if current >= increment {
                this.set(metric, current-increment);    
            } else {
                this.set(metric,0);
            }
        }   
        
        iter items() {
          for (key, val) in zip(counts.keys(), counts.values()) do
            yield(key, val);
        }
        
        proc size() {
            return this.counts.size;
        }
        
        proc total() {
            var count = 0;
            
            for item in this.items() {
                count += item[1];
            }
            
            return count; 
        }
    }
    
    proc exportAllMetrics() throws {        
        var metrics = new list(owned Metric?);

        for metric in getNumRequestMetrics() {
            metrics.pushBack(metric);
        }
        for metric in getResponseTimeMetrics() {
            metrics.pushBack(metric);
        }        
        for metric in getAvgResponseTimeMetrics() {
            metrics.pushBack(metric);
        }
        for metric in getTotalResponseTimeMetrics() {
            metrics.pushBack(metric);
        }
        for metric in getTotalMemoryUsedMetrics() {
            metrics.pushBack(metric);
        }
        for metric in getSystemMetrics() {
            metrics.pushBack(metric);
        }
        for metric in getServerMetrics() {
            metrics.pushBack(metric);
        }
        for metric in getAllUserRequestMetrics() {
            metrics.pushBack(metric);
        }
        for metric in getNumErrorMetrics() {
            metrics.pushBack(metric);
        }

        return metrics.toArray();
    }
   
    proc getUserRequestMetrics(userName: string) throws {
        return userMetrics.getUserRequestMetrics(userName);
    }

    proc getAllUserRequestMetrics() throws {
        return userMetrics.getPerUserNumRequestsPerCommandForAllUsersMetrics();
    }
 
    proc getServerMetrics() throws {
        var metrics: list(owned Metric?);
         
        for item in serverMetrics.items(){
            metrics.pushBack(new Metric(name=item[0], category=MetricCategory.SERVER, 
                                          value=item[1]));
        }

        return metrics;    
    }    

    proc getNumRequestMetrics() throws {
        var metrics = new list(owned Metric?);

        for item in requestMetrics.items() {
            metrics.pushBack(new Metric(name=item[0], 
                                      category=MetricCategory.NUM_REQUESTS,
                                      value=item[1]));
        }
        
        metrics.pushBack(new Metric(name='total', 
                                  category=MetricCategory.NUM_REQUESTS, 
                                  value=requestMetrics.total()));
        return metrics;
    }

    proc getNumErrorMetrics() throws {
        var metrics = new list(owned Metric?);

        for item in errorMetrics.items() {

            metrics.pushBack(new Metric(name=item[0],
                                        category=MetricCategory.NUM_ERRORS,
                                        value=item[1]));
        }

        metrics.pushBack(new Metric(name='total',
                                    category=MetricCategory.NUM_ERRORS,
                                    value=errorMetrics.total()));
        return metrics;
    }

    proc getPerUserNumRequestMetrics() throws {
        var metrics = new list(owned Metric?);

        for item in userMetrics.items() {
            metrics.pushBack(new Metric(name=item[0],
                                      category=MetricCategory.NUM_REQUESTS,
                                      value=item[1]));
        }

        metrics.pushBack(new Metric(name='total',
                                  category=MetricCategory.NUM_REQUESTS,
                                  value=requestMetrics.total()));
        return metrics;
    }


    proc getResponseTimeMetrics() throws {
        var metrics = new list(owned Metric?);

        for item in responseTimeMetrics.items() {
            metrics.pushBack(new Metric(name=item[0], 
                                      category=MetricCategory.RESPONSE_TIME,
                                      value=item[1]));
        }

        return metrics;
    }

    proc getAvgResponseTimeMetrics() throws {
        var metrics = new list(owned Metric?);

        for item in avgResponseTimeMetrics.items() {
            metrics.pushBack(new Metric(name=item[0], 
                                      category=MetricCategory.AVG_RESPONSE_TIME,
                                      value=item[1]));
        }

        return metrics;
    }

    proc getTotalResponseTimeMetrics() throws {
        var metrics = new list(owned Metric?);

        for item in totalResponseTimeMetrics.items() {
            metrics.pushBack(new Metric(name=item[0], 
                                      category=MetricCategory.TOTAL_RESPONSE_TIME,
                                      value=item[1]));
        }

        return metrics;
    }
    
    proc getTotalMemoryUsedMetrics() throws {
        var metrics = new list(owned Metric?);

        for item in totalMemoryUsedMetrics.items() {
            metrics.pushBack(new Metric(name=item[0], 
                                      category=MetricCategory.TOTAL_MEMORY_USED,
                                      value=item[1]));
        }

        return metrics;
    }

    proc getMaxLocaleMemory(loc) throws {
       if memMax:real > 0 {
           return memMax:real;
       } else {
           return loc.physicalMemory():real;
       }
    }
    proc getSystemMetrics() throws {
        var metrics = new list(owned Metric?);

        for loc in Locales {
            var used = memoryUsed():real;
            var total = getMaxLocaleMemory(loc);
            
            mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              'memoryUsed: %i physicalMemory: %i'.format(used,total));

            metrics.pushBack(new LocaleMetric(name="arkouda_memory_used_per_locale",
                             category=MetricCategory.SYSTEM,
                             locale_num=loc.id,
                             locale_name=loc.name,
                             locale_hostname = loc.hostname,
                             value=used):Metric);
            metrics.pushBack(new LocaleMetric(name="arkouda_percent_memory_used_per_locale",
                             category=MetricCategory.SYSTEM,
                             locale_num=loc.id,
                             locale_name=loc.name,
                             locale_hostname = loc.hostname,                             
                             value=used/total * 100):Metric);                            
        }
        return metrics;
    }
    
    proc getServerInfo() throws {
        var localeInfos = new list(owned LocaleInfo?); 

        for loc in Locales {
            var used = memoryUsed():int;
            var total = here.physicalMemory():int;

            var info = new LocaleInfo(name=loc.name,
                                      id=loc.id:string,
                                      hostname=loc.hostname,
                                      number_of_processing_units=loc.numPUs(),
                                      physical_memory=loc.physicalMemory():int,
                                      max_number_of_tasks=loc.maxTaskPar);   
            localeInfos.pushBack(info);                                                                                                                  
        }  
 
        var serverInfo = new ServerInfo(hostname=serverHostname, 
                                        version=arkoudaVersion,
                                        server_port=ServerPort,
                                        locales=localeInfos);
        return serverInfo;                            
    }
        
    class Metric {
        var name: string;
        var category: MetricCategory;
        var scope: MetricScope;
        var timestamp: dateTime;
        var value: real;
        
        proc init(name: string, category: MetricCategory, 
                                scope: MetricScope=MetricScope.GLOBAL, 
                                timestamp: dateTime=dateTime.now(), 
                                value: real) {
            this.name = name;
            this.category = category;
            this.scope = scope;
            this.timestamp = timestamp;
            this.value = value;
        }
    }
   
    class UserMetric : Metric {

        var user: string;

        proc init(name: string, category: MetricCategory,
                                scope: MetricScope=MetricScope.USER,
                                timestamp: dateTime=dateTime.now(),
                                value: real,
                                user: string) {

            super.init(name=name,
                       category = category,
                       scope = scope,
                       timestamp = timestamp,
                       value = value);
            this.user = user;
        }

    }

    class ArrayMetric : Metric {
        var cmd: string;
        var dType: DType;
        var size: int;
        
        proc init(name: string, 
                  category: MetricCategory, 
                  scope: MetricScope=MetricScope.GLOBAL, 
                  timestamp: dateTime=dateTime.now(), 
                  value: real,
                  cmd: string,
                  dType: DType,
                  size: int) {
              super.init(name=name,
                         category = category,
                         scope = scope,
                         timestamp = timestamp,
                         value = value
                        );
              this.cmd = cmd;
              this.dType = dType;
              this.size = size;
        }
    }

    class LocaleInfo {
        var name: string;
        var id: string;
        var hostname: string;
        var number_of_processing_units: int;
        var physical_memory: int;
        var max_number_of_tasks: int;
    }
    
    class ServerInfo {
        var hostname: string;
        var version: string;
        var server_port: int;
        var locales: [0..numLocales-1] owned LocaleInfo?;
        var number_of_locales: int;
        
        proc init(hostname: string, version: string, server_port: int,
                  locales) {
            this.hostname = hostname;
            this.version = version;
            this.server_port = server_port;
            this.locales = locales.toArray();
            this.number_of_locales = this.locales.size;
        }
    }

    class LocaleMetric : Metric {
        var locale_num: int;
        var locale_name: string;
        var locale_hostname: string;

        proc init(name: string, category: MetricCategory, 
                                scope: MetricScope=MetricScope.LOCALE, 
                                timestamp: dateTime=dateTime.now(), 
                                value: real, 
                                locale_num: int, 
                                locale_name: string, 
                                locale_hostname: string) {
            super.init(name=name, category=category, scope=scope, 
                                timestamp=timestamp, value=value);
            this.locale_num = locale_num;
            this.locale_name = locale_name;
            this.locale_hostname = locale_hostname;
        }
    }

    proc metricsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var category = msgArgs.getValueOf("category"):MetricCategory;
            
        mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            'category: %s'.format(category));
        var metrics: string;

        select category {
            when MetricCategory.ALL {
                metrics = formatJson(exportAllMetrics());
            }
            when MetricCategory.NUM_REQUESTS {
                metrics = formatJson(getNumRequestMetrics());
            }
            when MetricCategory.SERVER {
                metrics = formatJson(getServerMetrics());
            }
            when MetricCategory.SYSTEM {
                metrics = formatJson(getSystemMetrics());
            }
            when MetricCategory.SERVER_INFO {
                metrics = formatJson(getServerInfo());
            }
            when MetricCategory.TOTAL_MEMORY_USED {
                metrics = formatJson(getTotalMemoryUsedMetrics());            
            }
            when MetricCategory.AVG_RESPONSE_TIME {
                metrics = formatJson(getAvgResponseTimeMetrics());            
            }
            when MetricCategory.TOTAL_RESPONSE_TIME {
                metrics = formatJson(getTotalResponseTimeMetrics());            
            }
            when MetricCategory.NUM_ERRORS {
                metrics = formatJson(getNumErrorMetrics());
            }
            otherwise {
                throw getErrorWithContext(getLineNumber(),getModuleName(),getRoutineName(),
                      'Invalid MetricType', 'IllegalArgumentError');
            }
        }

        mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            'metrics %s'.format(metrics));
        return new MsgTuple(metrics, MsgType.NORMAL);        
    }

    use CommandMap;
    registerFunction("metrics", metricsMsg, getModuleName());
}
