"""Functions for loading in configurations from YAML files. Adapted from `selene_sdk`, which is in turn adapted from
`Pylearn2`.

Selene file:
https://github.com/FunctionLab/selene/blob/60e38a9a1b3cfde9bf7c9c28c0d72c2f53e1a63c/selene_sdk/utils/config.py

"""
import os
import warnings
import yaml


def load_configs(filename):
    """Load a YAML configuration from a file.

    Parameters
    ----------
    filename : str
        Name of the YAML file to load.

    Returns
    -------
    configs : dict
        Nested dictionary representation of the configs.
    """
    with open(filename, "r") as fin:
        content = "".join(fin.readlines())

    configs = yaml.load(content, Loader=yaml.SafeLoader)

    # Add some defaults
    if "output_dir" in configs.keys():
        output_dir = configs["output_dir"]
    else:
        output_dir = "mpra_tools_out"
        configs["output_dir"] = output_dir

    library_configs = configs["libraries"]
    for library in library_configs.keys():
        library_keys = library_configs[library].keys()
        if "write_output" not in library_keys:
            library_configs[library]["write_output"] = False

    # Change to the working directory, if needed
    if "work_dir" in configs.keys():
        os.chdir(configs["work_dir"])

    # Create the output directory
    #if os.path.exists(output_dir):
    #    warnings.warn(f"{output_dir} already exists, files may be overwritten.")
    #else:
    #    os.mkdir(output_dir)

    return configs
