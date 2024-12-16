FROM ajpotts/almalinux-chapel:1.0.1

# Set user to root
USER root


#   Download all the dependencies necessary for make install-deps
RUN git clone https://github.com/Bears-R-Us/arkouda.git && source ~/.bashrc && cd arkouda && make install-deps DEP_BUILD_DIR=/dep/build

#   Download all the python packages necessary for the pip install
COPY requirements.txt /root/
RUN mkdir /root/pip_deps && cd /root/pip_deps && python3 -m  pip download -r  /root/requirements.txt

ENTRYPOINT ["/bin/bash", "-l"]
