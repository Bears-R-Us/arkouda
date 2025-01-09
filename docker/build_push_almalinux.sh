#!/bin/bash

VERSION=1.0.0

docker build -t almalinux-chapel:$VERSION almalinux/
docker tag almalinux-chapel:$VERSION ajpotts/almalinux-chapel:$VERSION
docker push ajpotts/almalinux-chapel:$VERSION
