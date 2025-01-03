#!/bin/bash

VERSION=1.0.1
IMAGE_NAME=almalinux-chapel
REPO_NAME=ajpotts

docker build -t $IMAGE_NAME:$VERSION docker/$IMAGE_NAME/
#docker tag $IMAGE_NAME:$VERSION $REPO_NAME/$IMAGE_NAME:$VERSION
#docker push $REPO_NAME/$IMAGE_NAME:$VERSION
