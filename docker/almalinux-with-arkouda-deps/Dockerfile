FROM ajpotts/almalinux-chapel:1.0.0

# Set user to root
USER root


RUN git clone https://github.com/Bears-R-Us/arkouda.git && source ~/.bashrc && cd arkouda && make install-deps DEP_BUILD_DIR=/dep/build && python3 -m pip install -e .[dev]


ENTRYPOINT ["/bin/bash", "-l"]
