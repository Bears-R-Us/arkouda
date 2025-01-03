FROM ajpotts/almalinux-chapel:1.0.1

# Set user to root
USER root


#   Download all the dependencies necessary for make install-deps
RUN git clone -b 3933_failing_make_install-arrow_pt3 https://github.com/ajpotts/arkouda.git && source ~/.bashrc && cd arkouda && make install-deps DEP_BUILD_DIR=/dep/build

#   Download all the python packages necessary for the pip install
COPY requirements.txt /root/
RUN mkdir /root/pip_deps && cd /root/pip_deps && python3 -m  pip download -r  /root/requirements.txt

ENTRYPOINT ["/bin/bash", "-l"]
