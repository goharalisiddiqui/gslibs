"""
Common utility functions for gs-utils package.

This module provides general-purpose utility functions for colormap handling,
job monitoring, and command line argument parsing.
"""

from time import sleep
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

def wait_for_runs(runs):
    """
    Wait for all simulation runs to complete.

    Parameters
    ----------
    runs : list
        List of driver objects with is_job_running() method.
    """
    for run in runs:
        while run.is_job_running():
            sleep(5)
    
def parse_var(s):
    """
    Parse a key-value pair separated by '=' (reverse of shell arguments).

    Parameters
    ----------
    s : str
        String in the format 'key=value'.

    Returns
    -------
    tuple
        Tuple of (key, value) strings.

    Notes
    -----
    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split('=')
    key = items[0].strip() # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
    return (key, value)


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

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d
