"""
GS-Utils: A computational chemistry and materials science simulation toolkit.

This package provides drivers and utilities for running various computational
chemistry engines (LAMMPS, CP2K, ORCA) through SLURM job scheduler, along with
coordinate file manipulation and enhanced sampling support.

Modules
-------
drivers : Package containing simulation drivers
    base : Base driver class
    dftengines : DFT calculation drivers (CP2K, ORCA)
    mdengines : MD simulation drivers (LAMMPS, Gromacs)
    mlmodels : Machine learning model drivers (MACE, NequIP, MLColvar)
    colvarengines : Collective variable engines
    driverutils : SLURM utilities
utils : Package containing utility functions
    coordinates_io : Coordinate file I/O utilities
    plumed : PLUMED output file utilities
    common : Common utility functions
    filesystem : File system utilities
    orca : ORCA output parsing utilities
    log_read : Log file reading utilities
"""

__version__ = "0.1.0"
__author__ = "Gohar Ali Siddiqui"
__email__ = "goharalisiddiqui@gmail.com"

# Import main classes for convenience
from .drivers.base import base_driver
from .drivers.dftengines.cp2k import cp2k_driver
from .drivers.dftengines.orca import orca_driver
from .drivers.mdengines.lammps import lammps_driver
from .drivers.driverutils.slurm import slurm_driver

# Import utility functions
from .utils.coordinates_io import *
from .utils.plumed import plumed_outfile_reader, plumed_outfile_plotter
from .utils.common import get_colors, wait_for_runs, parse_var, parse_vars
from .utils.filesystem import backup_move

__all__ = [
    # Drivers
    'base_driver',
    'cp2k_driver',
    'orca_driver',
    'lammps_driver',
    'slurm_driver',
    # Coordinate I/O
    'any_to_ase',
    'lammpsdata_to_ase',
    'ase_to_lammpsdata',
    'pdb_to_ase',
    'ase_to_pdb',
    'xyz_to_ase',
    'ase_to_xyz',
    # PLUMED utilities
    'plumed_outfile_reader',
    'plumed_outfile_plotter',
    # Common utilities
    'get_colors',
    'wait_for_runs',
    'parse_var',
    'parse_vars',
    'backup_move',
]