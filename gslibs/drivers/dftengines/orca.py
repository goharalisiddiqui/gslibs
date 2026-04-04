"""
ORCA quantum chemistry driver for ab initio calculations.

This module provides a driver class for running ORCA quantum chemistry
calculations through the SLURM job scheduler.
"""

import argparse
import os
import numpy as np
from pymatgen.io.cp2k.outputs import Cp2kOutput

from gslibs.drivers.base import base_driver
from gslibs.utils.orca import orca_engrad_reader
from gslibs.utils.coordinates_io import *

class orca_driver(base_driver):
    """
    Driver for ORCA quantum chemistry calculations.

    Extends base_driver to provide functionality for running ORCA DFT
    calculations via SLURM job scheduler.

    Parameters
    ----------
    input_file : str
        Path to the ORCA input file.
    run_name : str
        Name identifier for the run.
    run_dir : str
        Directory where the calculation will be executed.
    verbose : bool, optional
        Enable verbose logging. Default is False.
    meta : dict, optional
        Metadata dictionary. Default is empty dict.
    """
    def __init__(self, input_file,
                 run_name: str,
                 run_dir: str,
                 verbose=False,
                 meta: dict = {}):
        """
        Initialize ORCA driver.

        Parameters
        ----------
        input_file : str
            Path to the ORCA input file.
        run_name : str
            Name identifier for the run.
        run_dir : str
            Directory where the calculation will be executed.
        verbose : bool, optional
            Enable verbose logging. Default is False.
        meta : dict, optional
            Metadata dictionary. Default is empty dict.
        """ 
        super().__init__(input_file, run_name, run_dir, verbose, meta)
        
        self.slurm_settings = {"partition": "zencloud", 
                               "time": "30:00:00", 
                               "mem": "60G",
                               "ntasks": 4,
                               "cpus-per-task": 8}
        self.runtime_env = {"OMP_NUM_THREADS": "8"}
        self.slurm_modules = ['orca/6.0.0']
        self.slurm_commands = []
        self.slurm_commands.append(f"mkdir rundir")
        self.slurm_commands.append(f"cd rundir")
        self.slurm_commands.append(f"cp ../* .")
        self.slurm_commands.append(f"$ORCA_HOME/orca {self.input_filename} >> ../orca.out")
    
    def write_final_singlepoint(self, dest, source_structure = "structure.xyz"):
        """
        Extract final single-point energy and forces from completed ORCA calculation.

        Parameters
        ----------
        dest : str
            Destination path for the final structure file.
        source_structure : str, optional
            Source structure filename. Default is "structure.xyz".

        Notes
        -----
        Reads ORCA output, extracts energy and forces, and writes final structure
        with properties to the specified destination file.
        """
        if '/' not in dest:
            dest = os.path.join(self.run_dir, dest)
        elif not dest.startswith("/"):
            raise ValueError(self.get_log_lines("Destination file must be an absolute path or a filename to be stored in the iteration directory itself."))
        self.check_status()
        if self.status != "COMPLETED":
            self.log_line("Job not finished. Skipping.")
            return
        if not os.path.exists(f"{self.run_dir}/orca.out"):
            self.log_line("Output file not found. Skipping.")
            return
        

        
        with open(f"{self.run_dir}/orca.out", 'r') as f:
            output = f.readlines()
        for line in output:
            if "ORCA TERMINATED NORMALLY" in line:
                break
        else:
            raise ValueError(self.get_log_lines("ORCA did not terminate normally. Exiting."))
            
            

        orca_engrad = orca_engrad_reader(f"{self.run_dir}/rundir/orca.engrad")
        forces = orca_engrad["Forces"]
        forces = np.array(forces)
        forces = -1 * forces # ORCA gives energy gradient, not forces
        
        final_energy = orca_engrad["Total Energy"]
            

        # Unit conversion
        forces = forces * 51.422086190832  # Hartree/bohr to eV/Angstrom
        final_energy = final_energy * 27.211386245988  # Hartree to eV
        
        
        
        # Extract the atoms and their positions
        if not source_structure.startswith("/"):
            source_structure = os.path.join(self.run_dir, source_structure)
        if not os.path.exists(source_structure):
            raise FileNotFoundError(self.get_log_lines(f"File {source_structure} not found."))
        structure = self.read_trajectory_file(source_structure)[0]

        structure.info["total_energy"] = final_energy
        structure.set_array("forces", forces)

        # Write the XYZ file
        if dest.endswith(".lammps"):
            output = ase_to_lammpsdata(structure)
        elif dest.endswith(".pdb"):
            output = ase_to_pdb(structure)
        elif dest.endswith(".xyz"):
            output = ase_to_xyz(structure)
            
        output = output.split("\n")
        
        dest_dir = os.path.dirname(dest)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        with open(dest, 'w') as f:
            lines = [line + "\n" for line in output if len(line) > 0]
            f.writelines(lines)
            self.log_line(f"Wrote final structure to {dest}")
        
        
        
        
        
            
            
        