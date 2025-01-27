#!/bin/bash


echo "This script is intended to be run from the arkouda project directory."

VERSION=1.0.1
IMAGE_NAME=almalinux-with-arkouda-deps
REPO_NAME=ajpotts

docker build -t $IMAGE_NAME:$VERSION docker/$IMAGE_NAME/
docker tag $IMAGE_NAME:$VERSION $REPO_NAME/$IMAGE_NAME:$VERSION
docker push $REPO_NAME/$IMAGE_NAME:$VERSION


