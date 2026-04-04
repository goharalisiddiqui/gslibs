"""
CP2K quantum chemistry driver for ab initio calculations.

This module provides a driver class for running CP2K quantum chemistry
calculations through the SLURM job scheduler.
"""

import os
import numpy as np
from pymatgen.io.cp2k.outputs import Cp2kOutput

from gslibs.drivers.base import base_driver
from gslibs.utils.coordinates_io import *

class cp2k_driver(base_driver):
    """
    Driver for CP2K quantum chemistry calculations.

    Extends base_driver to provide functionality for running CP2K DFT
    calculations via SLURM job scheduler.

    Parameters
    ----------
    input_file : str
        Path to the CP2K input file.
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
        Initialize CP2K driver.

        Parameters
        ----------
        input_file : str
            Path to the CP2K input file.
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
                               "mem": "32G",
                               "ntasks": 4,
                               "cpus-per-task": 8}
        self.runtime_env = {"OMP_NUM_THREADS": "8"}
        self.slurm_modules = ['cp2k/9.1']
        self.slurm_commands = [f"srun -n 4 cp2k.psmp -o cp2k.out -in {self.input_filename}"]
    
    def write_final_singlepoint(self, dest, source_structure = "structure.xyz"):
        """
        Extract final single-point energy and forces from completed CP2K calculation.

        Parameters
        ----------
        dest : str
            Destination path for the final structure file.
        source_structure : str, optional
            Source structure filename. Default is "structure.xyz".

        Notes
        -----
        Reads CP2K output, extracts energy and forces, applies unit conversion
        from Hartree to eV, and writes final structure with properties.
        """
        if '/' not in dest:
            dest = os.path.join(self.run_dir, dest)
        elif not dest.startswith("/"):
            raise ValueError(f"CP2K {self.run_name}: Destination file must be an absolute path or a filename to be stored in the iteration directory itself.")
        self.check_status()
        if self.status != "COMPLETED":
            print(f"CP2K {self.run_name}: Job not finished. Skipping.")
            return
        if not os.path.exists(f"{self.run_dir}/cp2k.out"):
            print(f"CP2K {self.run_name}: Output file not found. Skipping.")
            return
        output = Cp2kOutput(f"{self.run_dir}/cp2k.out")
        try:
            output.ran_successfully()
        except:
            print(f"CP2K {self.run_name}: Output file not found. Skipping.")
        output.parse_energies()
        output.parse_forces()
        # Extract the atoms and their positions
        if not source_structure.startswith("/"):
            source_structure = os.path.join(self.run_dir, source_structure)
        if not os.path.exists(source_structure):
            raise FileNotFoundError(f"CP2K {self.run_name}: File {source_structure} not found.")
        structure = self.read_trajectory_file(source_structure)[0]

        forces = np.squeeze(output.data["forces"])
        final_energy = output.final_energy

        # Unit conversion
        forces = forces * 51.422086190832  # Hartree/bohr to eV/Angstrom
        final_energy = final_energy * 27.211386245988  # Hartree to eV

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
            print(f"CP2K {self.run_name}: Wrote final structure to {dest}") if self.verbose else None
        
        
        
        
        
            
            
        