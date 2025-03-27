# External Tools

There are several external tools that are used in our current workflow and help us optimize development.
Many of the checks that automatically run on the Arkouda Github repo CI can be run locally.
Running these prior to pushing will speed up the approval process and help ensure that all code is organized and formatted in the same way.

## PyCharm External Tools

We encourage users to configure the following third-party command-line applications as external tools to run on their local machines in PyCharm before pushing to the Arkouda Github repo. They will also automatically be run in the runtime checks.

The current applications being used are `black`, `flake8`, and `isort`.

### Using Pip

Using `pip install`, the following applications can be downloaded to your arkouda directory.

```commandline
# navigate to the arkouda directory
cd <path_to_arkouda>/arkouda

# Download each of the following applications
pip install black==25.1.0
pip install flake8
pip install isort==5.13.2
```

### Add a local external tool 

1. In a new instance of Pycharm open the Settings.

2. Navigate to the Tools section.

3. Select External Tools.

4. Select the `+` to add a new external tool.

5. Existing tools can be right clicked or doubled clicked to be edited or changed. (Optional)

6. Select `OK` to save changes.

### Configuration for each tool

The following are the configurations and settings for each of the external tools.

#### `black`

```commandline
Name: black     Group: External Tools
Description: python formating

Tool Settings
    Program: $PyInterpreterDirectory$/black
    Augments: --line-length 105 "$FilePath$"
    Working Directory: $ProjectFileDir$

Advanced Options
    Synchronized Files after execution: Selected
    Open console for tool output: Selected
        Make console active on message in stdout: Unselected
        Make console active on message in stderr: Unselected
```

#### `flake8`

```commandline
Name: flake8     Group: External Tools
Description: pep8 formatting validation

Tool Settings
    Program: $PyInterpreterDirectory$/flake8
    Augments: "$FilePath$"
    Working Directory: $ProjectFileDir$

Advanced Options
    Synchronized Files after execution: Selected
    Open console for tool output: Selected
        Make console active on message in stdout: Unselected
        Make console active on message in stderr: Unselected
```

#### `isort`

```commandline
Name: isort     Group: External Tools
Description: sort/group imports

Tool Settings
    Program: $PyInterpreterDirectory$/isort
    Augments: "$FilePath$"
    Working Directory: $ProjectFileDir$

Advanced Options
    Synchronized Files after execution: Selected
    Open console for tool output: Selected
        Make console active on message in stdout: Unselected
        Make console active on message in stderr: Unselected
```

### Running the external tools

Each tool can be run from Pycharm by selecting the `Tools` drop down menu and 
then the `External Tools` drop down menu and selecting the desired tool.

## VSCode Chapel Language Syntax Highlighter

There is now a more current version of the the Chapel syntax highlighter that is available to be added as an extension to VSCode.

Open the Extensions tab in VS Code. Search for "Chapel Language" and install the extension titled "Chapel Language". (Identifier: chpl-hpe.chapel-vscode) 

In the details section of the description is the full set of instructions for installation. The description is also available to be viewed when selecting the extension from the Extensions tab in VSCode.

## Next Steps
You can now continue are ready to build the server! Follow the build instructions at [BUILD.md](BUILD.md).

Or you can return to the installation process at [LINUX_INSTALL.md](LINUX_INSTALL.md) or [MAC_INSTALL.md](MAC_INSTALL.md).
