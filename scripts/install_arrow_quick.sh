#!/bin/bash

DEP_BUILD_DIR=$1


# Check the necessary programs are installed
if ! command -v lsb_release 2>&1 >/dev/null
then
    printf "\nExiting.
    The program lsb_release could not be found.
    Please install lsb_release and try again, or use 'make install-arrow' instead.\n\n"
    exit 1
fi

#   get the OS, ubuntu, etc...
OS=$(lsb_release --id --short | tr 'A-Z' 'a-z')

#   If pop, replace with ubuntu
OS_FINAL=$(echo ${OS} | awk '{gsub(/pop/,"ubuntu")}1')

#   System release, such as "jammy" for "ubuntu jammy"
OS_CODENAME=$(lsb_release --codename --short)

#   System release, for example, 22 extracted from 22.04
OS_RELEASE=$(lsb_release -rs | cut -d'.' -f1)

if [[ $OS_FINAL == *"ubuntu"* ]] || [[ $OS_FINAL == *"debian"* ]]; then
	ARROW_LINK="https://apache.jfrog.io/artifactory/arrow/${OS_FINAL}/apache-arrow-apt-source-latest-${OS_CODENAME}.deb"
elif [[ $OS_FINAL == *"almalinux"* ]]; then
    ARROW_LINK="https://apache.jfrog.io/ui/native/arrow/${OS_FINAL}/${OS_RELEASE}/apache-arrow-release-latest.rpm"
elif [[ $OS_FINAL == *"centos-rc"* ]]; then
    ARROW_LINK="https://apache.jfrog.io/ui/native/arrow/centos-rc/9-stream/apache-arrow-release-latest.rpm"
fi

echo "Installing Apache Arrow/Parquet"
echo "from build directory: ${DEP_BUILD_DIR}"
mkdir -p ${DEP_BUILD_DIR}

#   If the BUILD_DIR does not contain the apache-arrow file, use wget to fetch it
if ! find ${DEP_BUILD_DIR} -name "apache-arrow*" -type f -print -quit | grep -q .; then
	cd ${DEP_BUILD_DIR} && wget ${ARROW_LINK}
fi

#   Now do the installs
if [[ $OS_FINAL == *"ubuntu"* ]] || [[ $OS_FINAL == *"debian"* ]]; then
    if [ "$EUID" -ne 0 ]; then
        cd $DEP_BUILD_DIR && sudo apt install -y -V ./apache-arrow*.deb
    else 
        cd $DEP_BUILD_DIR && apt install -y -V ./apache-arrow*.deb
    fi
elif [[ $OS_FINAL == *"almalinux"* ]]; then
    if [ "$EUID" -ne 0]; then
        cd $DEP_BUILD_DIR && sudo dnf install -y ./apache-arrow*.rpm
    else 
        cd $DEP_BUILD_DIR && dnf install -y ./apache-arrow*.rpm
    fi
else
    echo "make install-arrow-quick does not support ${OS}.  Please use make install-arrow instead."
fi


