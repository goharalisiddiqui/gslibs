"""
Coordinate file input/output utilities for various molecular structure formats.

This module provides functions to read and write molecular structures between
different file formats using the Atomic Simulation Environment (ASE) framework.
Supports formats including XYZ, PDB, LAMMPS data/dump, and Gromacs GRO.
"""

import os
from ase.io import read as ase_read
from ase.io.lammpsdata import write_lammps_data
from ase.io.proteindatabank import write_proteindatabank
from ase.io.gromacs import read_gromacs
from ase.io.extxyz import write_extxyz
from ase.io.xyz import write_xyz
from io import StringIO

def any_to_ase(file, extra_info=None):
    """
    Read any supported coordinate file format and convert to ASE atoms objects.

    Parameters
    ----------
    file : str
        Path to the coordinate file.
    extra_info : dict, optional
        Additional information for format-specific reading.

    Returns
    -------
    list of ase.Atoms or ase.Atoms
        Structure data as ASE atoms objects.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is not recognized.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} not found.")
    if file.endswith(".pdb"):
        return pdb_to_ase(file)
    elif file.endswith(".xyz"):
        return xyz_to_ase(file)
    elif file.endswith(".gro"):
        return gro_to_ase(file)
    elif file.endswith(".lammps"):
        return lammpsdata_to_ase(file, extra_info)
    elif file.endswith(".dump"):
        return lammpsdump_to_ase(file, extra_info)
    else:
        raise ValueError(f"File format not recognized for {file}")

def add_extra_information_to_ase(atoms, extra_info):
    """
    Add extra information like cell parameters and PBC to ASE atoms objects.

    Parameters
    ----------
    atoms : ase.Atoms or list of ase.Atoms
        ASE atoms object(s) to modify.
    extra_info : dict
        Dictionary containing 'cell' and/or 'pbc' information.

    Returns
    -------
    ase.Atoms or list of ase.Atoms
        Modified atoms object(s) with extra information.
    """
    if isinstance(atoms, list):
        for atom in atoms:
            if "cell" in extra_info and extra_info["cell"] is not None:
                atom.set_cell(extra_info["cell"])
            if "pbc" in extra_info and extra_info["pbc"] is not None:
                atom.set_pbc(extra_info["pbc"])
    else:
        if "cell" in extra_info and extra_info["cell"] is not None:
            atoms.set_cell(extra_info["cell"])
        if "pbc" in extra_info and extra_info["pbc"] is not None:
            atoms.set_pbc(extra_info["pbc"])
    return atoms

def lammpsdata_to_ase(lammps_file, extra_info=None, start=0, stop=0, step=1):
    """
    Read LAMMPS data file and convert to ASE atoms objects.

    Parameters
    ----------
    lammps_file : str
        Path to the LAMMPS data file.
    extra_info : dict, optional
        Extra information for reading.
    start : int, optional
        Starting frame index. Default is 0.
    stop : int, optional
        Stopping frame index. Default is 0 (read all).
    step : int, optional
        Step size for frame reading. Default is 1.

    Returns
    -------
    ase.Atoms or list of ase.Atoms
        Structure data as ASE atoms objects.

    Raises
    ------
    FileNotFoundError
        If the LAMMPS file does not exist.
    """
    if not os.path.exists(lammps_file):
        raise FileNotFoundError(f"File {lammps_file} not found.")
    frame = ase_read(lammps_file, format="lammps-data", index=f'{start}:{'' if stop == 0 else stop}:{step}', **extra_info if extra_info is not None else {})
    return frame

def ase_to_lammpsdata(atoms, extra_info=None):
    """
    Convert ASE atoms object to LAMMPS data file format string.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE atoms object to convert.
    extra_info : dict, optional
        Extra information including 'specorder', 'atom_style', 'units'.

    Returns
    -------
    str
        LAMMPS data file content as string.
    """
    ofile = StringIO()
    if extra_info is None:
        write_lammps_data(ofile, atoms)
    else:
        if "specorder" not in extra_info:
            print("UTILS: Warning: specorder not found in extra_info. Writing LAMMPS data file without spec_order. This can lead to discrepancies in the atom order.")
        write_lammps_data(ofile, atoms, **extra_info)
    return ofile.getvalue()

def lammpsdump_to_ase(traj_file, extra_info=None, start=0, stop=0, step=1):
    if not os.path.exists(traj_file):
        raise FileNotFoundError(f"File {traj_file} not found.")
    return ase_read(traj_file, format="lammps-dump-text", index=f'{start}:{'' if stop == 0 else stop}:{step}', **extra_info if extra_info is not None else {})

def pdb_to_ase(pdb_file, cell = None, pbc = None, start=0, stop=0, step=1):
    """
    Read PDB file and convert to ASE atoms objects.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file.
    cell : array_like, optional
        Unit cell parameters.
    pbc : array_like, optional
        Periodic boundary conditions.
    start : int, optional
        Starting frame index. Default is 0.
    stop : int, optional
        Stopping frame index. Default is 0 (read all).
    step : int, optional
        Step size for frame reading. Default is 1.

    Returns
    -------
    list of ase.Atoms
        Structure data as ASE atoms objects.

    Raises
    ------
    FileNotFoundError
        If the PDB file does not exist.
    """
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"File {pdb_file} not found.")
    structures = ase_read(pdb_file, format="proteindatabank", index=f'{start}:{'' if stop == 0 else stop}:{step}')
    return add_extra_information_to_ase(structures, {"cell": cell, "pbc": pbc})

def ase_to_pdb(atoms):
    ofile = StringIO()
    write_proteindatabank(ofile, atoms)
    return ofile.getvalue()

def xyz_to_ase(xyz_file, cell = None, pbc = None, start=0, stop=0, step=1):
    if not os.path.exists(xyz_file):
        raise FileNotFoundError(f"File {xyz_file} not found.")
    structures = ase_read(xyz_file, format="extxyz", index=f'{start}:{'' if stop == 0 else stop}:{step}')
    return add_extra_information_to_ase(structures, {"cell": cell, "pbc": pbc})

def ase_to_xyz(atoms, extxyz=True):
    """
    Convert ASE atoms object to XYZ format string.

    Parameters
    ----------
    atoms : ase.Atoms or list of ase.Atoms
        ASE atoms object(s) to convert.
    extxyz : bool, optional
        Whether to use extended XYZ format. Default is True.

    Returns
    -------
    str
        XYZ file content as string.
    """
    ofile = StringIO()
    if isinstance(atoms, list):
        if extxyz:
            write_extxyz(ofile, atoms)
        else:
            write_xyz(ofile, atoms)
    else:
        if extxyz:
            write_extxyz(ofile, [atoms])
        else:
            write_xyz(ofile, [atoms])
    return ofile.getvalue()


def gro_to_ase(gro_file, cell = None, pbc = None):
    if not os.path.exists(gro_file):
        raise FileNotFoundError(f"File {gro_file} not found.")
    structures = read_gromacs(gro_file)
    structures = [structures]
    return add_extra_information_to_ase(structures, {"cell": cell, "pbc": pbc})