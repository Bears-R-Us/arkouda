# External Integration

## Overview

Given the crucial Exploratory Data Analysis (EDA) role Arkouda fulfills in a variety of data science workflows, coupled with the popular trend of deploying machine learning (ML) and deep learning (DL) workloads to cloud environments, enabling Arkouda to be seamlessly integrated into cloud environments such as Kubernetes is an increasingly important use case.

## Design

Delivering integration with external systems such as Kubernetes requires three elements, all of which are encapsulated within the [ExternalIntegration](src/ExternalIntegration.chpl) module with the exception of one enum: 

1. Channel--implements logic for writing to external systems via export channels such as file systems and HTTP/S servers.
2. register/deregister--various register and deregister functions that register/deregister Arkouda with external systems via the corresponding Channel.
3. Enums--there are several enum classes that provide controlled vocabulary for external system and channel parameters.

### Channel

Channel derived classes override the Channel.write function to write the string payload parameter to an external system. For example, the HttpChannel class leverages the Chapel Curl module to write the JSON-formatted payloads used to register and deregister Arkouda with Kubernetes.

### register and deregister Functions

The ExternalIntegration register and deregister functions encapsulate logic to (1) generate the payload required to register and deregister Arkouda with/from external systems and (2) utilize a Channel object to deliver the register/deregister payload.

### Enums

The following enums provide controlled vocabulary to configure external integration:

1. SystemType--indicates the external system type, examples of which are KUBERNETES, CONSUL, and REDIS.
2. ChannelType--defines the type of channel used to integrate with an external systems, examples of which are FILE and HTTP.
3. ServiceEndpoint--indicates if the socket is for Arkouda client requests (for Arkouda server commands) or for metrics requests. 
4. HttpRequestType, HttpRequestFormat--enums used internally within the ExternalIntegration module to configure the HttpChannel in terms of request type (e.g., POST or PUT) and request format (e.g., TEXT or JSON).
5. Deployment--defined in the [ServerConfig](ServerConfig.chpl) module, the Deployment enum indicates whether Arkouda is deployed in a STANDARD environment (Slurm, bare metal) or KUBERNETES.

## Building Arkouda with External Integration Support

Since the ExternalIntegration module delegates HttpChannel registration logic to the Chaple Curl module, building Arkouda with ExternalIntegration requires the libcurl4-openssl-dev lib to be installed. For Debian and Ubuntu Linux distros, the install command is as follows:

```
sudo apt-get install libcurl4-openssl-dev
```

## Preparing External Systems for Integration

### Kubernetes 

The initial use case for Arkouda external integration is Kubernetes as described below. 

#### Required Files for Registering with Kubernetes

The Chapel Curl logic must use HTTPS to register/deregister with Kubernetes via the Kubernetes Rest API. Accordingly, a Kubernetes [ServiceAccount](https://kubernetes.io/docs/concepts/security/service-accounts/) and corresponding TLS token secret must be generated. The token and cacert file bound to the ServiceAccount sa are used to authenticate to the Kubernetes API. A ClusterRole or Role authorizing Pod, Service, and Endpoint creates/updates/deletes is bound to the Arkouda ServiceAccount. The cacert file must be deployed to all bare metal/slurm nodes to register Arkodua-on-Slurm with Kubernetes.

#### Create Kubernetes ServiceAccount

An example ServiceAccount is as follows. A key element is automountServiceAccountToken, which needs to be set to false in order to bind a long-lived token.

```
apiVersion: v1
kind: ServiceAccount
metadata:
  name: arkouda
automountServiceAccountToken: false
```

#### Create ServiceAccount Token

An example of secret encapsulating a ServiceAccount token is as follows. two key elements are the service-account.name and service-account-token Secret type:

```
apiVersion: v1
kind: Secret
metadata:
  name: arkouda-token
  annotations:
    kubernetes.io/service-account.name: arkouda
type: kubernetes.io/service-account-token
```

#### Output Token Data to Environment Variable

Once the token secret is created, the token string value used in the https command is created from the token secret is as follows:

```
export SECRET=arkouda-token
 
export TOKEN=$(kubectl get secret ${SECRET} -o json | jq -Mr '.data.token' | base64 -d)
```

The token is added to the Curl header submitted to the Kubernetes API HTTPS enpoint.

#### Output Token ca.crt Element to File

Once the token secret is created, the ca.crt string value used in the https command is created from the token secret is as follows:

```
export SECRET=arkouda-token
 
kubectl get secret ${SECRET} -o json | jq -Mr '.data["ca.crt"]' | base64 -d > ./ca.crt
```

The token string and ca.crt generated from the ServiceAccount token secret are used to authenticate to the Kubernetes API. An example Curl command is shown below using the TOKEN env variable and ca.crt file generated above:

```
export KUBE_URL=https://localhost:6443
curl $KUBE_URL/api/v1/namespaces/default/pods/ --header "Authorization: Bearer $TOKEN" --cacert ca.crt
```

#### Authorize read/write Access to Kubernetes Client API

With the Kubernetes arkouda ServiceAccount, TLS token, and ca.crt in place, create the RoleBinding or ClusterRoleBinding needed to authorize the arkouda ServiceAccount read/write access to the Kubernetes Client API.

```
kind: ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: arkouda
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: arkouda-service-pod
subjects:
- kind: ServiceAccount
  name: arkouda
  namespace: default

```

```
kubectl apply -f arkouda-role-binding.yaml
```

Important note: while this cluster role binding is valid, there may be some environments where it is desirable to narrow the arkouda user privileges.

## Use Cases and Examples

### Kubernetes

A stated above, integrating Arkouda with ML and DL workflows on Kubernetes is an increasingly important use case given the popularity of deploying ML/DL workflows to cloud environments generally, and Kubernetes specifically.

The registerWithKubernetes function generates the JSON blob containing either the standard (if Arkouda is deployed on Kubernetes) or external (Arkouda is deployed outside of Kubernetes on Slurm or bare-metal) service definition.

#### Registering Arkouda with Kubernetes

There are two inner functions that registerWithKubernetes delegates to:

1. registerAsInternalService--registers Arkouda as a standard Kubernetes service for Arkouda-on-Kubernetes deployments
2. registerAsExternalService--registers Arkouda-on-Slurm or bare metal--in other words, Arkouda deployed outside of Kubernetes--as an external Kubernetes service

The result of both registration inner functions is making Arkouda accessible to applications such as ML and DL workflows deployed in Kubernetes. 

#### Deregistering Arkouda from Kubernetes

The deregisterFromKubernetes function deletes the Kubernetes service and is triggered by the ak.shutdown() Arkouda client request.

#### Kubernetes Integration Configuration Parameters

The following environmental variables are required to configure Arkouda to register/deregister with Kubernetes:

1. K8S_HOST--the Kubernetes API connect string
2. NAMESPACE--Kubernetes namespace the service is deployed to
3. KEY_FILE--TLS key file corresponding to a Kubernetes user that has service create/read/delete privileges
4. CERT_FILE--TLS cert file corresponding to a Kubernetes user that has service create/read/delete privileges
5. EXTERNAL_SERVICE_PORT--port Arkouda will be accessible from
6. EXTERNAL_SERVICE_NAME--service name to access Arkouda

#### Kubernetes Internal Service Registration (Kubernetes)

Deployment of Arkouda-on-Kubernetes involves two Helm charts: one for the driver (locale0) pod, and one for 1..n locale (locale1...#locales-1) pods. The Helm installation process is detailed [here](https://github.com/hokiegeek2/arkouda/wiki/Arkouda-on-Docker-and-Kubernetes#deploying-multi-locale-arkouda-on-kubernetes). 

Note that the ExternalIntegration.externalSystem param is SystemType.KUBERNETES and the ServerConfig deployment param is Deployment.KUBERNETES (Arkouda is deployed on Kubernetes)

#### Kubernetes External Service Registration (Slurm)

An example Slurm BATCH file for an Arkouda instance that registers/deregisters with Kubernetes is shown below. Note that the ExternalIntegration.externalSystem param is SystemType.KUBERNETES and the deployment param is not specified because Slurm is considered a DEFAULT deployment type.

```
#!/bin/bash
#
#SBATCH --job-name=arkouda-3-node
#SBATCH --output=/tmp/arkouda.out
#SBATCH --mem=1024
#SBATCH --ntasks=3
#SBATCH --nodes=3
 
export CHPL_COMM_SUBSTRATE=udp
export GASNET_MASTERIP='server1'
export SSH_SERVERS='server1 server2 server3'
export GASNET_SPAWNFN=S

export NAMESPACE=arkouda
export K8S_HOST=https://localhost:6443 #result from kubectl cluster-info command
export EXTERNAL_SERVICE_NAME=arkouda-external
export EXTERNAL_SERVICE_PORT=5555
export KEY_FILE=/opt/arkouda/tls.key #on all slurm hosts
export CERT_FILE=/opt/arkouda/tls.crt #on all slurm hosts
export CACERT_FILE=/etc/kubernetes/ssl/kube-ca.pem #on slurm hosts

./arkouda_server -nl 3 --ExternalIntegration.systemType=SystemType.KUBERNETES \
                       --ServerDaemon.daemonTypes=ServerDaemonType.INTEGRATION
```

#### Kubernetes External Service Registration (Bare Metal)

An example bare metal deployment script for an Arkouda instance that registers/deregisters with Kubernetes is shown below. As is the case with the Arkouda-on-Slurm deployment, the ExternalIntegration.externalSystem param is SystemType.KUBERNETES and the deployment param is not specified because bare metal is considered a DEFAULT deployment type.

```
#!/bin/bash

export GASNET_MASTERIP='server1'
export SSH_SERVERS='server1 server2 server3'
export NAMESPACE=arkouda
export EXTERNAL_SERVICE_NAME=arkouda-external
export EXTERNAL_SERVICE_PORT=5555 
export K8S_HOST=https://localhost:6443 #result from kubectl cluster-info command
export KEY_FILE=/opt/arkouda.key #on all bare metal hosts
export CERT_FILE=/opt/arkouda.crt #on all bare metal hosts

./arkouda_server -nl 3 --ExternalIntegration.systemType=SystemType.KUBERNETES \
                       --ServerDaemon.daemonTypes=ServerDaemonType.INTEGRATION
```
