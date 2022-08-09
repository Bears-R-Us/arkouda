# External Integration

## Overview

Given the crucial Exploratory Data Analysis (EDA) role Arkouda fulfills in a variety of data science workflows, coupled with the popular trend of deploying machine learning (ML) and deep learning (DL) workloads to cloud environments, enabling Arkouda to be seamlessly integrated into cloud environments such as Kubernetes is an increasingly important use case.

## Design

Delivering integration with external systems such as Kubernetes requires four elements, all of which are encapsulated within the [ExternalIntegration](src/ExternalIntegration.chpl) module: 

1. Channel--implements logic for writing to external systems via export channels such as file systems and HTTP/S servers.
2. ChannelParams--encapsulates configuration parameters needed by Channel objects to connect to external systems such as file systems and HTTP/S servers.
3. register/deregister--various register and deregister functions that register/deregister Arkouda with external systems via the corresponding Channel.
4. Enums--there are several enum classes that provide controlled vocabulary for external system and channel parameters.

### Channel

Channel derived classes override the Channel.write function to write the string payload parameter to an external system. For example, the HttpChannel class leverages the Chapel Curl function to write the JSON-formatted payloads used to register and deregister Arkouda with/from Kubernetes, respectively:

### ChannelParams

The ChannelParams derived classes encapsulate the metadata required to connect and write to external systems via a Channel.

### register and deregister Functions

The ExternalIntegration register and deregister functions encapsulate logic to (1) generate the payload required to register and deregister Arkouda with/from external systems and (2) utilize a Channel object to deliver the register/deregister payload.

### Enums

The following enums provide controlled vocabulary to configure external integration:

1. SystemType--indicates the external system type, examples of which are KUBERNETES, CONSUL, and REDIS.
2. ChannelType--defines the type of channel used to integrate with an external systems, examples of which are FILE and HTTP.
3. ServiceType--indicates if Arkouda is deployed within or outside of the external system it is registering/deregistering with. For example, Arkouda registers itself as a Kubernetes service if it deployed in Kubernetes (ServiceType.INTERNAL) or outside (ServiceType.EXTERNAL) via Slurm or bare metal.
4. HttpRequestType, HttpRequestFormat--enums used internally within the ExternalIntegration module to configure the HttpChannel in terms of request type (e.g., POST or PUT) and request format (e.g., TEXT or JSON).

## Use Cases and Examples

### Kubernetes

#### Overview

A stated above, integrating Arkouda with ML and DL workflows on Kubernetes is an increasingly important use case given the popularity of deploying ML/DL workflows to cloud environments generally, and Kubernetes specifically.

The registerWithKubernetes function generates the JSON blob containing either the standard (if Arkouda is deployed internally on Kubernetes) or external (Arkouda is deployed outside of Kubernetes on Slurm or bare-metal) service definition.

#### Registering Arkouda with Kubernetes

There are two inner functions that registerWithKubernetes delegates to:

1. registerAsInternalService--registers Arkouda as a standard Kubernetes service for Arkouda-on-Kubernetes deployments
2. registerAsExternalService--registers Arkouda-on-Slurm or bare metal--in other words, Arkouda deployed outside of Kubernetes--as an external Kubernetes service

The result of both registration inner functions is making Arkouda accessible to applications such as ML and DL workflows deployed in Kubernetes. 

#### Deregistering Arkouda from Kubernetes

The deregisterFromKubernetes function deletes the internal or external Kubernetes service and is triggered by the ak.shutdown() Arkouda client request.

#### Kubernetes Registration/Deregistration Configuration Parameters

The following environmental variables are required to configure Arkouda to register/deregister with Kubernetes:

1. K8S_HOST--the Kubernetes API connect string
2. NAMESPACE--Kubernetes namespace the service is deployed to
3. KEY_FILE--TLS key file corresponding to a Kubernetes user that has service create/read/delete privileges
4. CERT_FILE--TLS cert file corresponding to a Kubernetes user that has service create/read/delete privileges
5. EXTERNAL_SERVICE_PORT--port Arkouda will be accessible from
6. EXTERNAL_SERVICE_NAME--service name to access Arkouda