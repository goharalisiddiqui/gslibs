"""
ORCA quantum chemistry output file parsing utilities.

This module provides functions to read and parse ORCA output files,
particularly the .engrad files containing energies and gradients.
"""

import os

def orca_engrad_reader(engrad_file):
    """
    Read and parse ORCA .engrad file containing energies and gradients.

    Parameters
    ----------
    engrad_file : str
        Path to the ORCA .engrad output file.

    Returns
    -------
    dict
        Dictionary containing parsed data with keys:
        - 'Total Energy': float, final energy in atomic units
        - 'Forces': list, atomic forces in Eh/bohr
        - 'Number of atoms': int, number of atoms
        - Other fields as found in the file

    Raises
    ------
    AssertionError
        If the engrad file does not exist.
    """
    assert os.path.exists(engrad_file), f"Output file {engrad_file} not found."
    with open(engrad_file, 'r') as f:
        engrad = f.readlines()
    fields = {}
    key = None
    data = []
    i = 0
    while True:
        if engrad[i].startswith("#"):
            if key is not None:
                if len(data) == 1:
                    if len(data[0]) == 1:
                        data = data[0][0]
                    else:
                        data = data[0]
                elif all(len(x) == 1 for x in data):
                    data = [x[0] for x in data]
                fields[key] = data
                key = None
                data = []
            while engrad[i] == "#\n":
                i += 1
            key = engrad[i][2:-1]
            i += 1
            while engrad[i] == "#\n":
                i += 1
        data.append([float(x) for x in engrad[i][:-1].split()])
        i += 1
        if i == len(engrad):
            if key is not None:
                if len(data) == 1:
                    if len(data[0]) == 1:
                        data = data[0][0]
                    else:
                        data = data[0]
                elif all(len(x) == 1 for x in data):
                    data = [x[0] for x in data]
                fields[key] = data
            break
    
    fields_final = {}
    ## Post processing
    for key, value in fields.items():
        if key == "Number of atoms":
            fields_final[key] = int(value)
        elif "The current total energy" in key:
            fields_final["Total Energy"] = value
        elif key == "The current gradient in Eh/bohr":
            assert len(value) == fields["Number of atoms"] * 3
            fields_final['Forces'] = [value[i:i+3] for i in range(0, len(value), 3)]
        else:
            fields_final[key] = value
    
    return fields_final