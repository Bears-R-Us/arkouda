FROM ajpotts/almalinux-chapel:1.0.1

# Set user to root
USER root


#   Download all the dependencies necessary for make install-deps
#  TODO: fix specific commit
#  RUN cd /opt && git clone https://github.com/Bears-R-Us/arkouda.git  && git checkout abfb13bfe7931c07c2dec4e814955c59551a2d98
RUN cd /opt && git clone -b 4341_Upgrade_to_arrow_19.0.1 https://github.com/ajpotts/arkouda.git
RUN cd /opt/arkouda && source ~/.bashrc && make install-deps DEP_BUILD_DIR=/dep/build

#   Download all the python packages necessary for the pip install
COPY requirements.txt /root/
RUN mkdir /root/pip_deps && cd /root/pip_deps && python3 -m  pip download -r  /root/requirements.txt

ENTRYPOINT ["/bin/bash", "-l"]
