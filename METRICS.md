# Arkouda Metrics Generation and Export

## Overview

Arkouda generates measurement, count, system, and user metrics and makes them available for export via a dedicated metrics socket. The export semantics and data structures are modeled for metrics export to Prometheus where (1) the ArkoudaMetrics server polls the Arkouda metrics socket per a configurable polling interval and (2) the Prometheus exporter scrapes the ArkoudaMetrics service endpoint. The result is time series data stream for each metric that can be loaded into Prometheus for short-term storage and TimescaleDB for long-term storage.

## Metrics Generation and Export: The MetricsMsg Module

The [MetricsMsg](src/MetricsMsg.chpl) module contains logic and data structures to generate and cache metrics as well as generate JSON blobs to encapsulate all metrics to be exported. Specifically, the MetricsMsg module contains the following code required to generate and export metrics from Arkouda:

1. Increment/decrement counter metrics and capture measure metrics
2. Encapsulated counts and measurements in CounterTable or MeasurementTable Chapel Maps
3. Encapsulate metric data and metadata within a Metric, UserMetric, and LocaleMetric objects
4. Export Arkouda metrics as JSON blobs generated from Metric, UserMetric, and LocaleMetric objects

### Metrics Generation

#### Measurement Metrics

Measurement metrics are generated in the MeasurementsTable class:

```
    proc get(metric: string) : int {
        if !this.measurements.contains(metric) {
            this.measurements.add(metric,0.0);
            return 0;
        } else {
            return try! this.measurements.getValue(metric);
        }
    }   
        
    proc set(metric: string, measurement: real) {
        this.measurements.addOrReplace(metric, measurement);
    }
```

#### Count Metrics

Count metrics are captured in the Counter Table:

```
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
```

#### UserMetrics

The UserMetrics contains logic to increment counts such as total number of requests and number of requests per command:

```
    proc incrementPerUserRequestMetrics(userName: string, metricName: string, increment: int=1) {
        this.incrementNumRequestsPerCommand(userName,metricName,increment);
        this.incrementTotalNumRequests(userName,increment);
    }
        
    proc incrementNumRequestsPerCommand(userName: string, cmd: string, increment: int=1) {
        var userMetrics : borrowed CounterTable = this.getUserMetrics(this.users.getUser(userName));
        userMetrics.increment(cmd,increment);
    }   
    
    proc incrementTotalNumRequests(userName: string,increment: int=1) {
        var userMetrics : borrowed CounterTable = this.getUserMetrics(this.users.getUser(userName));
        userMetrics.increment('total',increment);
    }     
```

#### System Metrics

System metrics are generated in the getSystemMetrics function:

```
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
```

### Metrics Export

All metrics are exported as a JSON blob via the following logic:

```
    proc exportAllMetrics() throws {        
        var metrics = new list(owned Metric?);

        metrics.extend(getNumRequestMetrics());
        metrics.extend(getResponseTimeMetrics());
        metrics.extend(getSystemMetrics());
        metrics.extend(getServerMetrics());

        for userMetric in getAllUserRequestMetrics() {
            metrics.append(userMetric: owned Metric);
        }

        return metrics.toArray();
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
    
    proc getResponseTimeMetrics() throws {
        var metrics = new list(owned Metric?);

        for item in responseTimeMetrics.items() {
            metrics.append(new Metric(name=item[0], 
                                      category=MetricCategory.RESPONSE_TIME,
                                      value=item[1]));
        }

        return metrics;
    }
    
    proc getServerMetrics() throws {
        var metrics: list(owned Metric?);
         
        for item in serverMetrics.items(){
            metrics.append(new Metric(name=item[0], category=MetricCategory.SERVER, 
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
```

The MetricsMsg module is integrated into the Arkouda server side workflow within the MetricsServerDaemon class located in the [ServerDaemon](src/ServerDaemon) module. 

## Enabling Metrics Capture and Export

The arkouda_server startup command that enables metrics capture and export has two variants

### Standard Arkouda Deployment

In situations where Arkouda is not registered with an external system such as Kubernetes, the launch command is as follows. Note METRICS_SERVICE_PORT only has to be set if the port cannot be the default value of 5556.

```
export METRICS_SERVICE_PORT=6556

./arkouda_server -nl 3 --memTrack=true --ServerDaemon.daemonTypes=ServerDaemonType.DEFAULT,ServerDaemonType.METRICS
```

### Integration-Enabled Arkouda Deployment

In situations where Arkouda is registered with an external system, in this case Kubernetes, the ServerDaemonType is switched to INTEGRATION and extra environment variables are added as needed.

```
export NAMESPACE=arkouda
export EXTERNAL_SERVICE_NAME=arkouda-external
export EXTERNAL_SERVICE_PORT=5555
export METRICS_SERVICE_NAME=arkouda-metrics
export METRICS_SERVICE_PORT=5556
export K8S_HOST=https://localhost:6443 #result from kubectl cluster-info command
export KEY_FILE=/opt/arkouda/arkouda.key #on all bare metal hosts
export CERT_FILE=/opt/arkouda/arkouda.crt #on all bare metal hosts

./arkouda_server -nl 3 --memTrack=true --ExternalIntegration.systemType=SystemType.KUBERNETES \
                       --ServerDaemon.daemonTypes=ServerDaemonType.INTEGRATION,ServerDaemonType.METRICS \
                       --logLevel=LogLevel.DEBUG
```

## Arkouda Prometheus Exporter

The [arkouda_metrics_exporter](https://github.com/Bears-R-Us/arkouda-contrib/tree/main/arkouda_metrics_exporter) project is located in the [arkouda-contrib](https://github.com/Bears-R-Us/arkouda-contrib) repository. The arkouda_metrics_exporter [metrics](https://github.com/Bears-R-Us/arkouda-contrib/blob/main/arkouda_metrics_exporter/client/arkouda_metrics_exporter/metrics.py) module contains a simple Arkouda Prometheus exporter composed of the ArkoudaMetrics class and an http_server that serves as the Prometheus scrape target. The Arkouda implementation is based upon an excellent [example](https://trstringer.com/quick-and-easy-prometheus-exporter/) developed and documented by Thomas Stringer.

### Core Logic of Arkouda Prometheus Exporter

The Python [prometheus_client](https://github.com/prometheus/client_python) library contains the core functionality required to deliver a Prometheus exporter. The ArkoudaMetrics fetch() method makes a call to MetricsMsg w/ the 'ALL' parameter, meaning that all metrics will be returned to the client and are prepared for Prometheus scrape requests.

```
    def fetch(self) -> None:
        metrics = json.loads(
            client.generic_msg(cmd="metrics", args=str(MetricCategory.ALL)),
            object_hook=self.asMetric,
        )

        if len(metrics) > 0:
            self._assignTimestamp(metrics)

        for metric in metrics:
            self._updateMetric[metric.category](metric)
            logger.debug("UPDATED METRIC {}".format(metric))
```

Within the asMetric method the incoming JSON blobs emitted from Arkouda are converted to Prometheus data structures:

```
    def asMetric(self, value: Dict[str, Union[float, int]]) -> Metric:
        scope = MetricScope(value["scope"])
        labels: Optional[List[Label]]

        if scope == MetricScope.LOCALE:
            labels = [
                Label("locale_name", value=value["locale_name"]),
                Label("locale_num", value=value["locale_num"]),
                Label("locale_hostname", value=value["locale_hostname"]),
            ]
            return Metric(
                name=str(value["name"]),
                category=MetricCategory(value["category"]),
                scope=MetricScope(value["scope"]),
                value=value["value"],
                timestamp=parser.parse(cast(str, value["timestamp"])),
                labels=labels,
            )

        elif scope == MetricScope.USER:
            user = cast(str, value["user"])
            labels = [Label("user", value=user)]
            return UserMetric(
                name=str(value["name"]),
                category=MetricCategory(value["category"]),
                scope=MetricScope(value["scope"]),
                value=value["value"],
                timestamp=parser.parse(cast(str, value["timestamp"])),
                user=user,
                labels=labels,
            )
        else:
            labels = None
            return Metric(
                name=str(value["name"]),
                category=MetricCategory(value["category"]),
                scope=MetricScope(value["scope"]),
                value=value["value"],
                timestamp=parser.parse(cast(str, value["timestamp"])),
                labels=labels,
            )
```

Within the main loop (1) the HTTP server that constitutes the scrape endpoint starts up and (2) the run_metrics_loop method periodically retrieves metric data from Arkouda.

```
def main():
    """Main entry point"""

    pollingInterval = int(os.getenv("POLLING_INTERVAL_SECONDS", "5"))
    exportPort = int(os.getenv("EXPORT_PORT", "5080"))

    metrics = ArkoudaMetrics(
        exportPort=exportPort,
        pollingInterval=pollingInterval
    )
    start_http_server(exportPort)
    metrics.run_metrics_loop()
```

## Arkouda Prometheus Exporter Deployment

### Bare Metal

To run on bare metal, run the following shell script:

```
#!/bin/bash
  
export METRICS_SERVICE_NAME=<kubernetes external service name or arkouda server hostname>
export METRICS_SERVICE_PORT=5556
export POLLING_INTERVAL_SECONDS=5
export EXPORT_PORT=5080
export ARKOUDA_SERVER_NAME=arkouda-ventura-metrics-exporter

python -c 'from arkouda.monitoring import main; main()'
    _         _                   _       
   / \   _ __| | _____  _   _  __| | __ _ 
  / _ \ | '__| |/ / _ \| | | |/ _` |/ _` |
 / ___ \| |  |   < (_) | |_| | (_| | (_| |
/_/   \_\_|  |_|\_\___/ \__,_|\__,_|\__,_|
                                          

Client Version: v2022.07.28+37.g631d5f32.dirty
Starting Prometheus scrape endpoint
Started Prometheus scrape endpoint
```
