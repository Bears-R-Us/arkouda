# Windows (WSL2)

It's possible to install arkouda on Windows using Windows Subsystem for Linux (WSL2).
The general strategy here is to use a Linux terminal on WSL to launch the server.
If you are going to try this route, we suggest using WSL-2 with the latest Ubuntu.
There are a number of tutorials available online such as [MicroSoft's](https://docs.microsoft.com/en-us/windows/wsl/install-win10)

Key installation points:

- Make sure to use WSL2
- The latest Ubuntu from the MS app store
- Don't forget to create a user account and password as part of the Linux install

Once configured you can follow the basic [Linux Install](LINUX_INSTALL.md)
for installing Chapel & Arkouda.  We also recommend installing Anaconda for windows.

**Notes:** 
* By default, wsl VM is allocated 8 Gb of memory. If you want to increase this for your
    system, you will need to
    [add a `.wslconfig` file](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#wslconfig). The contents should
    looks something like the following
    ```
    [wsl2]
    memory=12GB # Limits VM memory in WSL 2 to 12 GB 
    ```
    To verify, restart wsl and run a command like `top` or `htop`

* When running `make` to build Chapel while using WSL, pathing issues to library dependencies are common. In most cases, a symlink pointing to the correct location or library will fix these errors.

    An example of one of these errors found while using Chapel 1.31.0 and Ubuntu 20.04 LTS with WSL is:
    
    ```bash
    ../../../bin/llvm-tblgen: error while loading shared libraries: libtinfow.so.6: cannot open shared object file: No such file or directory
    ```
    
    This error can be fixed by the following command:
    
    ```bash
    sudo ln -s /lib/x86_64-linux-gnu/libtic.so.6.2 /lib/x86_64-linux-gnu/libtinfow.so.6
    ```
    
    The general plan is to compile & run the `arkouda-server` process from a Linux terminal on WSL and then either connect
    to it with the python client using another Linux terminal running on WSL _or_ using the Windows Anaconda-Powershell.
    
    If running an IDE you can use either the Windows or Linux version, however, you may need to install an X-window system
    on Windows such as VcXsrv, X410, or an alternative.  Follow the setup instructions for whichever one you choose, but
    keep in mind you may need to update your Windows firewall to allow the Xserver to connect.  Also, on the Linux side of
    the house we found it necessary to add 
    
    ```bash
    export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2; exit;}'):0.0
    ```
    
    to our `~/.bashrc` file to get the display correctly forwarded.
