"""
Common utility functions for gs-utils package.

This module provides general-purpose utility functions for colormap handling,
job monitoring, and command line argument parsing.
"""

from time import sleep
import inspect
import numpy as np

def get_colors(n: int, colormap):
    """
    Generate a list of colors from a matplotlib colormap.

    Parameters
    ----------
    n : int
        Number of colors to generate.
    colormap : matplotlib colormap
        Colormap to sample from.

    Returns
    -------
    list
        List of color values from the colormap.
    """
    cmr = colormap.resampled(n)
    return [cmr(i) for i in np.linspace(0, 1, n)]

def wait_for_runs(runs, sleep_time: int = 5):
    """
    Wait for all simulation runs to complete.

    Parameters
    ----------
    runs : list
        List of driver objects with is_job_running() method.
    sleep_time : int, optional
        Time to sleep between checks (default is 5).
    """
    for run in runs:
        while run.is_job_running():
            sleep(sleep_time)

def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary.

    Parameters
    ----------
    items : list or None
        List of strings in 'key=value' format.

    Returns
    -------
    dict
        Dictionary of parsed key-value pairs.
    """
    d = {}
    def parse_var(s):
        items = s.split('=')
        key = items[0].strip() # we remove blanks around keys, as is logical
        if len(items) > 1:
            # rejoin the rest:
            value = '='.join(items[1:])
        return (key, value)

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d

def parse_slice(st):
    """
    Parse a string into a slice object.
    The string should be in the format "start:stop:step".
    """
    if st == "":
        return slice(None)
    if ":" not in st:
        return slice(int(st), None, None)
    parts = st.split(":")
    if len(parts) == 1:
        return slice(int(parts[0]), None, None)
    elif len(parts) == 2:
        return slice(int(parts[0]), int(parts[1]), None)
    elif len(parts) == 3:
        return slice(int(parts[0]), int(parts[1]), int(parts[2]))
    else:
        raise ValueError(f"Invalid slice format: {st}")

def recursive_update(d, u):
    '''
    Recursively update dictionary d with values from dictionary u.
    If a key in u corresponds to a dictionary in d, the function will 
    recursively update that dictionary instead of overwriting it.
    
    Parameters:
    d (dict): The dictionary to be updated.
    u (dict): The dictionary with updates to apply to d.
    
    Returns:
    None: The function updates d in place and does not return anything. 
    '''
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            recursive_update(d[k], v)
        else:
            d[k] = v

def get_required_init_args(cls):
    '''
    Get the list of required argument names for the __init__ method of a class.
    This function uses the inspect module to analyze the signature of the 
    __init__ method and returns a list of argument names that are required 
    (i.e., those that do not have default values and are not 'self').
    
    Parameters:
    cls (type): The class for which to retrieve the required __init__ argument 
    names.
    
    Returns:
    list: A list of required argument names.
    '''
    sig = inspect.signature(cls.__init__)
    # Skip 'self' and arguments with default values
    return [
        name for name, param in sig.parameters.items()
        if name != 'self' and param.default is param.empty and param.kind in 
            (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    ]

def get_init_args(cls):
    '''
    Get the list of all argument names for the __init__ method of a class.
    This function uses the inspect module to analyze the signature of the 
    __init__ method and returns a list of all argument names (including those 
    with default values, but excluding 'self').
    
    Parameters:
    cls (type): The class for which to retrieve the __init__ argument names.
    
    Returns:
    list: A list of all argument names.
    '''
    sig = inspect.signature(cls.__init__)
    return [
        name for name, param in sig.parameters.items()
        if name != 'self' and param.kind in 
            (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    ]