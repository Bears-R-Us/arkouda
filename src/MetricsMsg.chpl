module MetricsMsg {
    use ServerConfig;
    use Reflection;
    use ServerErrors;
    use Logging;    
    use List;
    use IO;
    use Map;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use Message;
    use Memory.Diagnostics;
    use ArkoudaDateTimeCompat;

    enum MetricCategory{ALL,NUM_REQUESTS,RESPONSE_TIME,SYSTEM,SERVER,SERVER_INFO};
    enum MetricScope{GLOBAL,LOCALE,REQUEST,USER};

    private config const logLevel = ServerConfig.logLevel;
    const mLogger = new Logger(logLevel);

    var metricScope = ServerConfig.getEnv(name='METRIC_SCOPE',default='MetricScope.REQUEST');
    
    var serverMetrics = new CounterTable();
    
    var requestMetrics = new CounterTable();
    
    var responseTimeMetrics = new MeasurementTable();

    var users = new Users();
    
    var userMetrics = new UserMetrics();

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
                return try! users.getValue(name);
            }
        }

        proc getUserNames() {
            return this.users.keys();
        }

        proc getUsers() {
            return this.users.values();
        }
    }

    class UserMetrics {
        var metrics = new map(keyType=User,valType=shared CounterTable);
        var users = new Users();

        proc getUserMetrics(user: User) {
            if this.metrics.contains(user: User) {
                return try! this.metrics.getValue(user);
            } else {
                var userMetrics = new shared CounterTable();
                this.metrics.add(user, userMetrics);
                return userMetrics;
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
                metrics.append(new UserMetric(name=metric,
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
                    metrics.append(metric);
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

    class MeasurementTable {
        var measurements = new map(string, real);
    
        proc get(metric: string) : int {
            if !this.measurements.contains(metric) {
                this.measurements.add(metric,0.0);
                return 0;
            } else {
                return try! this.measurements.getValue(metric);
            }
        }   
        
        proc set(metric: string, measurement: real) {
            this.measurements.addOrSet(metric, measurement);
        }
        
        proc items() {
            return this.measurements.items();
        }
        
        proc size() {
            return this.measurements.size;
        }
    
    }

    class CounterTable {
        var counts = new map(string, int);
       
        proc get(metric: string) : int {
            if !this.counts.contains(metric) {
                this.counts.add(metric,0);
                return 0;
            } else {
                return try! this.counts.getValue(metric);
            }
        }   
        
        proc set(metric: string, count: int) {
            this.counts.addOrSet(metric,count);
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
        
        proc items() {
            return this.counts.items();
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
            metrics.append(metric);
        }
        for metric in getResponseTimeMetrics() {
            metrics.append(metric);
        }
        for metric in getSystemMetrics() {
            metrics.append(metric);
        }
        for metric in getServerMetrics() {
            metrics.append(metric);
        }
        for metric in getAllUserRequestMetrics() {
            metrics.append(metric);
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
            metrics.append(new Metric(name=item[0], category=MetricCategory.SERVER, 
                                          value=item[1]));
        }

        return metrics;    
    }    

    proc getNumRequestMetrics() throws {
        var metrics = new list(owned Metric?);

        for item in requestMetrics.items() {
            metrics.append(new Metric(name=item[0], 
                                      category=MetricCategory.NUM_REQUESTS,
                                      value=item[1]));
        }
        
        metrics.append(new Metric(name='total', 
                                  category=MetricCategory.NUM_REQUESTS, 
                                  value=requestMetrics.total()));
        return metrics;
    }

    proc getPerUserNumRequestMetrics() throws {
        var metrics = new list(owned Metric?);

        for item in userMetrics.items() {
            metrics.append(new Metric(name=item[0],
                                      category=MetricCategory.NUM_REQUESTS,
                                      value=item[1]));
        }

        metrics.append(new Metric(name='total',
                                  category=MetricCategory.NUM_REQUESTS,
                                  value=requestMetrics.total()));
        return metrics;
    }


    proc getResponseTimeMetrics() throws {
        var metrics = new list(owned Metric?);

        for item in responseTimeMetrics.items() {
            metrics.append(new Metric(name=item[0], 
                                      category=MetricCategory.RESPONSE_TIME,
                                      value=item[1]));
        }

        return metrics;
    }

    proc getSystemMetrics() throws {
        var metrics = new list(owned Metric?);

        for loc in Locales {
            var used = memoryUsed():real;
            var total = loc.physicalMemory():real;
            
            mLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              'memoryUsed: %i physicalMemory: %i'.format(used,total));

            metrics.append(new LocaleMetric(name="arkouda_memory_used_per_locale",
                             category=MetricCategory.SYSTEM,
                             locale_num=loc.id,
                             locale_name=loc.name,
                             locale_hostname = loc.hostname,
                             value=used):Metric);
            metrics.append(new LocaleMetric(name="arkouda_percent_memory_used_per_locale",
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
            localeInfos.append(info);                                                                                                                  
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
        var timestamp: datetime;
        var value: real;
        
        proc init(name: string, category: MetricCategory, 
                                scope: MetricScope=MetricScope.GLOBAL, 
                                timestamp: datetime=datetime.now(), 
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
                                timestamp: datetime=datetime.now(),
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
                                timestamp: datetime=datetime.now(), 
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
                metrics = "%jt".format(exportAllMetrics());
            }
            when MetricCategory.NUM_REQUESTS {
                metrics = "%jt".format(getNumRequestMetrics());
            }
            when MetricCategory.SERVER {
                metrics = "%jt".format(getServerMetrics());
            }
            when MetricCategory.SYSTEM {
                metrics = "%jt".format(getSystemMetrics());
            }
            when MetricCategory.SERVER_INFO {
                metrics = "%jt".format(getServerInfo());
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
