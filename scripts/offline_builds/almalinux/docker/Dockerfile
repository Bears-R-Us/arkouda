FROM ajpotts/almalinux-chapel:1.0.1

# Set user to root
USER root

RUN git clone -b offline_build_scripts_pr2 https://github.com/ajpotts/arkouda.git

RUN mkdir -p /root/new_dir

#   Download all the python packages necessary for the pip install
COPY requirements.txt /root/
COPY build.tar /root/
COPY pip_deps.tar /root/

RUN cd /root/new_dir && tar -xvf /root/build.tar 
RUN cd /root/new_dir && tar -xvf /root/pip_deps.tar
RUN echo "export ARROW_DEPENDENCIES_SOURCE_DIR=/root/new_dir/build/arrow_dependencies" >> ~/.bashrc
RUN dnf install -y vim

ENTRYPOINT ["/bin/bash", "-l"]
