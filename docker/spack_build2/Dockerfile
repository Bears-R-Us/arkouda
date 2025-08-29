# docker build command
# cls;cd /home/mwmarde/arkouda-util/Docker && docker build --build-arg RELEASE_VERSION=2025.07.03 -t arkouda_v2025.07.03:0.13 .
# cls;cd /home/mwmarde/arkouda-util/Docker && docker build -t arkouda_v2025.07.03:0.13 .
# cd /home/mwmarde/arkouda-util/Docker &&  docker run -it arkouda_v2025.07.03:0.13 

# Use AlmaLinux as the base image
FROM almalinux:9

# Maintainer and metadata
LABEL maintainer="ajpotts@datacraft-solutions.com"
LABEL description="Arkouda development image with Spack and Chapel pinned to compatible versions"

# Arkouda release to be build
ARG RELEASE_VERSION
ENV RELEASE_VERSION=${RELEASE_VERSION:+@${RELEASE_VERSION}}
RUN echo "RELEASE_VERSION: ${RELEASE_VERSION}"

# Set non-interactive installation
ENV TZ=UTC
ENV DEBIAN_FRONTEND=noninteractive

# Use Bash as the shell for all RUN statements
SHELL ["/bin/bash", "-c"]

# ---------------------------
# Add aliases directly to .bashrc (or create one if it doesn't exist)
# ---------------------------
RUN echo "alias cls='clear'" >> /root/.bashrc && \
    echo "alias e='exit'" >> /root/.bashrc && \
    echo "alias h='history'" >> /root/.bashrc && \
    echo "alias ll='ls -alF'" >> /root/.bashrc && \
    echo "alias lrt='ls -lrt --color=auto'" >> /root/.bashrc

# ---------------------------
# Install system dependencies
# --allowerasing helps resolve curl-minimal and other conflicts
# ---------------------------
RUN dnf install -y --allowerasing \
    git gcc gcc-c++ make cmake lsof \
    python3 python3-devel python3-pip \
    curl tar gzip wget which \
    findutils gawk patch file less hostname unzip \
    diffutils procps-ng libcurl-devel bzip2 \
    && dnf clean all

# ---------------------------
# Clone Spack and set environment
# ---------------------------
WORKDIR /opt
RUN git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git spack-develop

ENV SPACK_ROOT=/opt/spack-develop
ENV PATH=$SPACK_ROOT/bin:$PATH
ENV CHPL_GMP=bundled

# ---------------------------
# Initialize Spack and find available compilers
# ---------------------------
RUN . $SPACK_ROOT/share/spack/setup-env.sh && spack compiler find



#################################################################################



RUN source spack-develop/share/spack/setup-env.sh && spack compilers

# ---------------------------
# Apply in-place sed patches to the Arkouda package.py
#   1. Pin chapel@2.4.0:2.4.99 +hdf5 +zmq
#   2. Disable lz4 in arrow (~lz4)
# ---------------------------
# Replace full chapel dependency line
#RUN export ARKOUDA_PACKAGE_FILE=$(find /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages | grep /arkouda/package.py) && \
#    sed -i 's|version("master", branch="master")|version("main", branch="main")|' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2024.12.06", sha256="92ca11319a9fdeeb8879afbd1e0c9c1b1d14aa2496781c1481598963d3c37b46")' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2025.01.13", sha256="bb53bab92fedf43a47aadd9195eeedebe5f806d85887fa508fb5c69f2a4544ea")' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2025.07.03", sha256="eb888fac7b0eec6b4f3bfa0bfe14e5c8f15b449286e84c45ba95c44d8cd3917a")' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2025.08.20", sha256="3e305930905397ff3a7a28a5d8cc2c9adca4194ca7f6ee51f749f427a2dea92c")' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i 's|depends_on("chapel@[^"]*", type=(.*))|depends_on("chapel@2.4.0:2.4.99 +hdf5 +zmq", type=("build", "link", "run", "test"))|' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i 's|arrow +parquet +snappy +zlib +brotli +bz2 +lz4 +zstd|arrow +parquet +snappy +zlib +brotli +bz2 ~lz4 +zstd|' \
#    $ARKOUDA_PACKAGE_FILE

COPY arkouda-package.py /opt/arkouda-package.py

RUN export ARKOUDA_PACKAGE_FILE=$(find /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages | grep /arkouda/package.py) \
    &&  mv /opt/arkouda-package.py $ARKOUDA_PACKAGE_FILE


# ---------------------------
# Install Arkouda with patched Spack package
# ---------------------------
RUN source spack-develop/share/spack/setup-env.sh && spack install arkouda${RELEASE_VERSION}



# ---------------------------
# Expose the default arkouda_server port
# ---------------------------
#EXPOSE 5555

# ---------------------------
# Apply in-place sed patches to the py_arkouda/package.py
# ---------------------------
#RUN export ARKOUDA_CLIENT_PACKAGE_FILE=$(find /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages | grep /py_arkouda/package.py) && \
#    sed -i 's|version("master", branch="master")|version("main", branch="main")|' \
#    $ARKOUDA_CLIENT_PACKAGE_FILE && \
#    sed -i 's|maintainers("arezaii")|maintainers("ajpotts", "arezaii")|' \
#    $ARKOUDA_CLIENT_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2024.12.06", sha256="92ca11319a9fdeeb8879afbd1e0c9c1b1d14aa2496781c1481598963d3c37b46")' \
#    $ARKOUDA_CLIENT_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2025.01.13", sha256="bb53bab92fedf43a47aadd9195eeedebe5f806d85887fa508fb5c69f2a4544ea")' \
#    $ARKOUDA_CLIENT_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2025.07.03", sha256="eb888fac7b0eec6b4f3bfa0bfe14e5c8f15b449286e84c45ba95c44d8cd3917a")' \
#    $ARKOUDA_CLIENT_PACKAGE_FILE \
#    sed -i '30i \ \ \ \ version("2025.08.20", sha256="3e305930905397ff3a7a28a5d8cc2c9adca4194ca7f6ee51f749f427a2dea92c")' \
#    $ARKOUDA_CLIENT_PACKAGE_FILE

COPY py_arkouda-package.py /opt/py_arkouda-package.py
RUN export ARKOUDA_CLIENT_PACKAGE_FILE=$(find /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages | grep /py_arkouda/package.py) && \
    mv /opt/py_arkouda-package.py $ARKOUDA_CLIENT_PACKAGE_FILE



# ---------------------------
# Install arkouda python front end
# Wheel files will be saved here /opt/wheels
# So the project can be installed offline later
# ---------------------------
#RUN dnf -y update && dnf -y install gfortran
#RUN spack compiler find
#RUN spack install py-arkouda ^py-setuptools@68.2.2 ^openblas@0.3.28
#RUN spack install py-arkouda${RELEASE_VERSION}

# ---------------------------
# Copy a simple test script
# ---------------------------




WORKDIR /root

#ENTRYPOINT ["/bin/bash", "-l"]

CMD ["/bin/bash"]







