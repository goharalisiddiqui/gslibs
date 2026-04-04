"""
PLUMED output file reading and plotting utilities.

This module provides functions to read and visualize data from PLUMED output files,
which are commonly used in enhanced sampling molecular dynamics simulations.
"""

import os
import pandas as pd

import matplotlib.pyplot as plt

def plumed_outfile_reader(plumed_outfile, columns = None, ignore_list = [], column_match = None, vstime = True):
    """
    Read PLUMED output file and extract specified columns.

    Parameters
    ----------
    plumed_outfile : str
        Path to the PLUMED output file.
    columns : list, optional
        List of column names to extract. Default extracts all columns.
    ignore_list : list, optional
        List of column names to ignore. Default is empty.
    column_match : str, optional
        String pattern to match in column names.
    vstime : bool, optional
        Whether to use time as index. Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the extracted data.

    Raises
    ------
    FileNotFoundError
        If the PLUMED output file does not exist.
    ValueError
        If file format is invalid or specified columns are not found.
    """
    for token in ['#!', 'FIELDS']:
        if token not in ignore_list:
            ignore_list.append(token)
    if not os.path.exists(plumed_outfile):
        raise FileNotFoundError(f"File {plumed_outfile} not found.")
    with open(plumed_outfile, 'r') as f:
        line = f.readline()
    if "FIELDS" not in line:
        raise ValueError(f"File {plumed_outfile} does not contain the correct header. Its probably not a PLUMED output file.")
    fields = line.split()[3:]
    if columns != None:
        if not all([col in fields for col in columns]):
            not_found = [col for col in columns if col not in fields]
            raise ValueError(f"Columns {not_found} not found in the PLUMED output file. Columns found are {fields}")
        columns = [field for field in fields if field in columns]
    else:
        columns = fields
    if ignore_list != None:
        columns = [col for col in columns if col not in ignore_list]
    if column_match != None:
        columns = [col for col in columns if column_match in col]
    column_positions = [fields.index(col) + 1 for col in columns]
    if vstime:
        column_positions = [0] + column_positions
        data = pd.read_csv(plumed_outfile, sep='\\s+', comment='#', header=None, usecols=column_positions, index_col=0)
    else:
        data = pd.read_csv(plumed_outfile, sep='\\s+', comment='#', header=None, usecols=column_positions, index_col=False)
    data.columns = columns

    return data

def plumed_outfile_plotter(plumed_outfile, columns = None, ignore_list = None, column_match = None, title = None, vstime = True):
    """
    Plot data from PLUMED output file.

    Parameters
    ----------
    plumed_outfile : str
        Path to the PLUMED output file.
    columns : list, optional
        List of column names to plot.
    ignore_list : list, optional
        List of column names to ignore.
    column_match : str, optional
        String pattern to match in column names.
    title : str, optional
        Title for the plot.
    vstime : bool, optional
        Whether to plot vs time. Default is True.

    Notes
    -----
    Creates a multi-panel plot with each specified column in a separate subplot.
    Special handling for phi/psi angles with point markers.
    """

    data = plumed_outfile_reader(plumed_outfile, columns = columns, ignore_list = ignore_list, column_match = column_match, vstime = vstime)
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 2*len(columns)), squeeze=False)


    for i, col in enumerate(columns):
        if col in ['phi', 'psi']:
            axes[i][0].plot(data.index, data[col], '+')

        else:
            axes[i][0].plot(data.index, data[col])
        axes[i][0].set_ylabel(col)
        axes[i][0].grid()
        # if col in ['opes.zed']:
        #     axes[i].set_ylim(0.5, 1.5)

    if title != None:
        fig.suptitle(title)

    axes[i][0].set_xlabel('Time (ps)')

    plt.tight_layout()
    plt.show()
