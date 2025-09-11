FROM ghcr.io/bears-r-us/almalinux-chapel:latest

# Set user to root
USER root


#   Download all the dependencies necessary for make install-deps
#  Fix specific commit https://github.com/Bears-R-Us/arkouda/pull/4342
RUN cd /opt && git clone https://github.com/Bears-R-Us/arkouda.git  && cd /opt/arkouda && git checkout 33298be6ab0d75125ee6513e1d5375485c70676c
RUN cd /opt/arkouda && source ~/.bashrc && make install-deps DEP_BUILD_DIR=/dep/build

#   Download all the python packages necessary for the pip install
COPY requirements.txt /root/
RUN mkdir /root/pip_deps && cd /root/pip_deps && python3 -m  pip download -r  /root/requirements.txt

ENTRYPOINT ["/bin/bash", "-l"]
