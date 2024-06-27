module ExternalIntegration {
    use Curl;
    use URL;
    use Reflection;
    use FileIO;
    use Logging;
    use ServerConfig;
    use ServerErrors;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const eiLogger = new Logger(logLevel, logChannel);

    private config const systemType = SystemType.NONE;
    
    /*
     * libcurl C constants required to configure the Curl core
     * of HttpChannel objects.
     */
    extern const CURLOPT_VERBOSE:CURLoption;
    extern const CURLOPT_USERNAME:CURLoption;
    extern const CURLOPT_PASSWORD:CURLoption;
    extern const CURLOPT_USE_SSL:CURLoption;
    extern const CURLOPT_SSLCERT:CURLoption;
    extern const CURLOPT_SSLKEY:CURLoption;
    extern const CURLOPT_KEYPASSWD:CURLoption;
    extern const CURLOPT_SSLCERTTYPE:CURLoption;
    extern const CURLOPT_CAPATH:CURLoption;
    extern const CURLOPT_CAINFO:CURLoption;
    extern const CURLOPT_URL:CURLoption;
    extern const CURLOPT_HTTPHEADER:CURLoption;
    extern const CURLOPT_POSTFIELDS:CURLoption;
    extern const CURLOPT_CUSTOMREQUEST:CURLoption;  
    extern const CURLOPT_FAILONERROR:CURLoption;
    extern const CURLINFO_RESPONSE_CODE:CURLoption;
    extern const CURLOPT_SSL_VERIFYPEER:CURLoption;

    /*
     * Enum specifies the type of external system Arkouda will integrate with.
     */
    enum SystemType{KUBERNETES,REDIS,CONSUL,NONE};

    /*
     * Enum describing the type of channel used to write to an
     * external system.
     */
    enum ChannelType{STDOUT,FILE,HTTP};
       
    /*
     * Enum specifies if the service endpoint is the Arkouda client or metrics
     * socket 
     */
    enum ServiceEndpoint{ARKOUDA_CLIENT,METRICS};
    
    /*
     * Enum specifies the request type used to write to an external system 
     * via HTTP.
     */
    enum HttpRequestType{POST,PUT,PATCH,DELETE};

    /*
     * Enum specifies the request format used to write to an external system 
     * via HTTP.
     */
    enum HttpRequestFormat{TEXT,JSON,MULTIPART};    

    /*
     * Retrieves the host ip address of the locale 0 arkouda_server process, which is
     * useful for registering Arkouda with cloud environments such as Kubernetes.
     */
    proc getConnectHostIp() throws {
        var hostip: string;
        on Locales[0] {
            var ipString = getLineFromFile('/etc/hosts',getConnectHostname());
            try {
                var splits = ipString.split();
                hostip = splits[0]:string;
                hostip.split();
            } catch (e: Error) {
                throw new IllegalArgumentError(
                         "invalid hostname -> ip address entry in /etc/hosts %?".format(
                                               e));
            }
        }
        return hostip;
    }

    /*
     * Base class defining the Arkouda Channel interface consisting of a
     * write method that writes a payload to an external system.
     */
    class Channel {
        proc write(payload : string) throws {
            throw new owned Error("All derived classes must implement write");
        }
    }

    /*
     * The FileChannel class writes a payload out to a file, either by appending
     * or overwriting an existing file or creating and writing to a new file.
     */
    class FileChannel : Channel {
        var path: string;
        var append: bool;
       
        proc init(path: string, append: bool) {
            super.init();
            this.path = path;
            this.append = append;
        }
        
        override proc write(payload: string) throws {
            if append {
                appendFile(path, payload);
            } else {
                writeToFile(path, payload);
            }
        }
    }
    
    /*
     * The HttpChannel class writes a payload out to an HTTP endpoint
     * in a configurable format via a configurable request type.
     */
    class HttpChannel : Channel {
        var url: string;
        var requestType: HttpRequestType;
        var requestFormat: HttpRequestFormat;
 
        proc init(url: string, requestType: HttpRequestType, requestFormat: HttpRequestFormat) {
            super.init();

            this.url = url;
            this.requestType = requestType;
            this.requestFormat = requestFormat;
        }

        proc configureChannel(channel) throws {
            if logLevel == LogLevel.DEBUG {
                Curl.setopt(channel, CURLOPT_VERBOSE, true);
            }
        }
        
        proc generateHeader(channel) throws {
            var args = new Curl.slist();
            var format = this.requestFormat;
            select(format) {     
                when HttpRequestFormat.JSON {
                    args.append("Accept: application/json");
                    if this.requestType == HttpRequestType.PATCH {
                        args.append('Content-Type: application/json-patch+json');
                    } else {
                        args.append("Content-Type: application/json");    
                    }               
                }     
                when HttpRequestFormat.TEXT {
                    args.append("Accept: text/plain");
                    args.append("Content-Type: text/plain; charset=UTF-8");
                } 
                otherwise {
                    throw new Error("Unsupported HttpFormat");
                }
                
            }

            Curl.curl_easy_setopt(channel, CURLOPT_HTTPHEADER, args);  
            return args;
        }
        
        /*
         * Writes the payload out to an HTTP/S endpoint in a format specified
         * by the requestFormat instance attribute via the request type 
         * specified in the requestType instance attribute.
         */
        override proc write(payload: string) throws {
            var curl = Curl.curl_easy_init();

            Curl.curl_easy_setopt(curl, CURLOPT_URL, this.url);
           
            this.configureChannel(curl);
            
            Curl.curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1);
            
            var args = generateHeader(curl);

            Curl.curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload);
            Curl.curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, this.requestType:string);

            eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "Configured HttpChannel for type %s format %s".format(
                      this.requestType, this.requestFormat));

            eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "Executing Http request with payload %s".format(payload));

            var ret = Curl.curl_easy_perform(curl);
            
            if ret == 0 {
                eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "Successfully executed Http request with payload %s".format(payload));
            } else {
                if ret == 22 {
                    throw getErrorWithContext(getLineNumber(),getRoutineName(),getModuleName(),
                       "error in HTTP request %s".format(payload),
                       "ExternalSystemError");

                } else { 
                    throw getErrorWithContext(getLineNumber(),getRoutineName(),getModuleName(),
                       "request with payload %s returned error code %i".format(payload,ret),
                       "ExternalSystemError");
                }
            }

            args.free();
            Curl.curl_easy_cleanup(curl);     
            
            eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "Closed HttpChannel");      
        }
    }    


    /*
     * The HttpChannel class writes a payload out to an HTTPS endpoint
     * in a configurable format via a configurable request type.
     */
    class HttpsChannel: HttpChannel {
        var caCert: string;
        var token: string;

        proc init(url: string, requestType: HttpRequestType, requestFormat: HttpRequestFormat, 
                  caCert: string, token: string) {
            super.init(url, requestType, requestFormat);
            this.caCert = caCert;
            this.token = token;
        }

        /**
         * Overridden proc adds token TLS configuration
         */
        override proc configureChannel(channel) throws {
            super.configureChannel(channel);
            
            Curl.setopt(channel, CURLOPT_USE_SSL, true);
            Curl.curl_easy_setopt(channel, CURLOPT_USE_SSL, true);
            Curl.curl_easy_setopt(channel, CURLOPT_CAINFO, this.caCert);
            Curl.curl_easy_setopt(channel, CURLOPT_SSL_VERIFYPEER, 0);

            eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "Configured channel for ssl and CA cert %s".format(this.caCert));
        }

        /**
         * Overridden proc adds TLS token to the HTTPS header
         */
        override proc generateHeader(channel) throws {
            var args = super.generateHeader(channel);

            args.append("Authorization: Bearer %s".format(getEnv("SSL_TOKEN")));
            Curl.curl_easy_setopt(channel, CURLOPT_HTTPHEADER, args);  
            return args;
        }
    }
  
    /*
     * Registers Arkouda with Kubernetes by creating a Kubernetes Service--and an Endpoints 
     * if Arkouda is deployed outside of Kubernetes--to enable service discovery of Arkouda 
     * from applications deployed within Kubernetes.
     */
    proc registerWithKubernetes(appName: string, serviceName: string, 
                                         servicePort: int, targetServicePort: int) throws {
        if deployment == Deployment.KUBERNETES {
            registerAsInternalService(appName, serviceName, servicePort, targetServicePort);
        } else {
            registerAsExternalService(serviceName, servicePort, targetServicePort);
        }

        proc generateEndpointCreateUrl() : string throws {
            var k8sHost = ServerConfig.getEnv('K8S_HOST');
            var namespace = ServerConfig.getEnv('NAMESPACE');
            return '%s/api/v1/namespaces/%s/endpoints'.format(k8sHost,namespace);
        }
    
        proc generateEndpointUpdateUrl() : string throws {
            var k8sHost = ServerConfig.getEnv('K8S_HOST');
            var namespace = ServerConfig.getEnv('NAMESPACE');
            var name = ServerConfig.getEnv('ENDPOINT_NAME');
            return '%s/api/v1/namespaces/%s/endpoints/%s'.format(k8sHost,namespace,name);
        }

        proc generateServiceCreateUrl() : string throws {
            var k8sHost = ServerConfig.getEnv('K8S_HOST');
            var namespace = ServerConfig.getEnv(name='NAMESPACE',default='default');
            return '%s/api/v1/namespaces/%s/services'.format(k8sHost,namespace);
        }

        proc registerAsInternalService(appName: string, serviceName: string, servicePort: int, 
                                       targetPort: int) throws {
            var serviceUrl = generateServiceCreateUrl();
            var servicePayload = "".join('{"apiVersion": "v1","kind": "Service","metadata": ',
                                         '{"name": "%s"},"spec": {"ports": [{"port": %i,' ,
                                         '"protocol": "TCP","targetPort": %i}],"selector":',
                                         ' {"app":"%s"}}}').format(
                                         serviceName,
                                         servicePort,
                                         targetPort,
                                         appName);

            eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                     "Registering internal service via payload %s and url %s".format(
                                         servicePayload,serviceUrl));

            var channel = new HttpsChannel(url=serviceUrl,
                                           requestType=HttpRequestType.POST,
                                           requestFormat=HttpRequestFormat.JSON,
                                           caCert=ServerConfig.getEnv('CACERT_FILE'),
                                           token=ServerConfig.getEnv('SSL_TOKEN'));

            channel.write(servicePayload);
        
            eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                     "Registered internal service via payload %s and url %s".format(
                                         servicePayload,serviceUrl));  
        }

        /*
         * Registers Arkouda with Kubernetes by creating a Kubernetes Service and 
         * Endpoints which together enable service discovery of an Arkouda instance 
         * deployed outside of Kubernetes from applications deployed within Kubernetes.
         */        
        proc registerAsExternalService(serviceName: string, servicePort: int, 
                                                               serviceTargetPort: int) throws {
            // Create Kubernetes Service
            var serviceUrl = generateServiceCreateUrl();
            var servicePayload = "".join('{"apiVersion": "v1","kind": "Service","metadata": ',
                                             '{"name": "%s"},"spec": {"ports": [{"port": %i,',
                                             '"protocol": "TCP","targetPort": %i}]}}').format(
                                    serviceName,
                                    servicePort,
                                    serviceTargetPort);
            eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                     "Registering external service via payload %s and url %s".format(
                                         servicePayload,serviceUrl));

            var channel = new HttpsChannel(url=serviceUrl,
                                           requestType=HttpRequestType.POST,
                                           requestFormat=HttpRequestFormat.JSON,
                                           caCert=ServerConfig.getEnv('CACERT_FILE'),
                                           token=ServerConfig.getEnv('SSL_TOKEN'));

            channel.write(servicePayload);
            eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                     "Registered external service via payload %s and url %s".format(
                                         servicePayload,serviceUrl));       
            
            // Create Kubernetes Endpoints  
            var endpointUrl = generateEndpointCreateUrl();                                                                                     
            var endpointPayload = "".join('{"kind": "Endpoints","apiVersion": "v1",',
                                          ' "metadata": {"name": "%s"}, "subsets": ',
                                          '[{"addresses": [{"ip": "%s"}],"ports": ',
                                          '[{"port": %i, "protocol": "TCP"}]}]}').format(
                                                serviceName,
                                                getConnectHostIp(),
                                                servicePort);
        
            channel = new HttpsChannel(url=endpointUrl,
                                       requestType=HttpRequestType.POST,
                                       requestFormat=HttpRequestFormat.JSON,
                                       caCert=ServerConfig.getEnv('CACERT_FILE'),
                                       token=ServerConfig.getEnv('SSL_TOKEN'));

            eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                     "Registering endpoint via payload %s and url %s".format(
                                         endpointPayload,endpointUrl));

            channel.write(endpointPayload);      
            eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                     "Registered endpoint via payload %s and endpointUrl %s".format(
                                         endpointPayload,endpointUrl)); 
        }
    }
    
    /*
     * Removes the Kubernetes Service and, if applicable, Endpoints that compose the
     * service endpoint that enables access to Arkouda deployed within or outside of 
     * Kubernetes from applications deployed within Kubernetes
     */
    proc deregisterFromKubernetes(serviceName: string) throws {
        proc generateServiceDeleteUrl(serviceName: string) throws {
            var k8sHost = ServerConfig.getEnv('K8S_HOST');
            var namespace = ServerConfig.getEnv('NAMESPACE');   
            return '%s/api/v1/namespaces/%s/services/%s'.format(k8sHost,namespace,serviceName);
        }
        
        var url = generateServiceDeleteUrl(serviceName);
        var channel = new HttpsChannel(url=url,
                                       requestType=HttpRequestType.DELETE,
                                       requestFormat=HttpRequestFormat.JSON,
                                       caCert=ServerConfig.getEnv('CACERT_FILE'),
                                       token=ServerConfig.getEnv('SSL_TOKEN'));
        channel.write('{}');
        eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                     "Deregistered service %s from Kubernetes via url %s".format(serviceName, 
                                                                                 url));
    }
    
    proc getKubernetesRegistrationParameters(serviceEndpoint: ServiceEndpoint) throws {
        var serviceName: string;
        var servicePort: int;
        var targetServicePort: int;

        if serviceEndpoint == ServiceEndpoint.METRICS {
            serviceName = ServerConfig.getEnv('METRICS_SERVICE_NAME');
            servicePort = ServerConfig.getEnv('METRICS_SERVICE_PORT', 
                                               default='5556'):int;
            servicePort = ServerConfig.getEnv('METRICS_SERVICE_TARGET_PORT', 
                                               default='5556'):int;
        } else {
            serviceName = ServerConfig.getEnv('EXTERNAL_SERVICE_NAME');
            serviceName = ServerConfig.getEnv('EXTERNAL_SERVICE_NAME');
            servicePort = ServerConfig.getEnv('EXTERNAL_SERVICE_PORT', 
                                               default='5555'):int;
            targetServicePort = ServerConfig.getEnv('EXTERNAL_SERVICE_TARGET_PORT', 
                                                     default='5555'):int;
        }

        return (serviceName,servicePort,targetServicePort);
    } 

    proc getKubernetesDeregisterParameters(serviceEndpoint: ServiceEndpoint) throws {
        if serviceEndpoint == ServiceEndpoint.METRICS {
            return ServerConfig.getEnv('METRICS_SERVICE_NAME');
        } else {
            return ServerConfig.getEnv('EXTERNAL_SERVICE_NAME');
        }
    }
    
    /*
     * Registers Arkouda with an external system on startup, defaulting to none.
     */
    proc registerWithExternalSystem(appName: string, endpoint: ServiceEndpoint) throws {                                                       
        select systemType {
            when SystemType.KUBERNETES {
                var params: (string,int,int) = getKubernetesRegistrationParameters(endpoint); 

                registerWithKubernetes(appName, params(0), params(1), params(2));
                eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "Registered Arkouda with Kubernetes");
            }
            otherwise {
                eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "Did not register Arkouda with any external systems");            
            }
        }
    }
        
    /*
     * Deregisters Arkouda from an external system upon receipt of shutdown command
     */
    proc deregisterFromExternalSystem(endpoint: ServiceEndpoint) throws {
        var serviceName = getKubernetesDeregisterParameters(endpoint); 

        select systemType {
            when SystemType.KUBERNETES {
                deregisterFromKubernetes(serviceName);
                eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "Deregistered service %s from Kubernetes".format(serviceName));
            }
            otherwise {
                eiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "Did not deregister Arkouda from any external system");
            }
        }
    }
}
