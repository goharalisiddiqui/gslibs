import os
import numpy as np

from gslibs.utils.plumed import *
from gslibs.utils.coordinates_io import *

from gslibs.drivers.base import base_driver

class gromacs_driver(base_driver):
    def __init__(self, input_file: str, 
                 run_name: str,
                 run_dir: str,
                 verbose=False,
                 meta: dict = {},
                 slurm_settings: dict = {},
                 ):
        super().__init__(input_file, run_name, run_dir, verbose, meta)

        self.slurm_settings = {"partition": "zencloud", 
                               "time": "3:00:00", 
                               "mem": "20G",
                               "ntasks": "1",
                               "cpus-per-task": "8"
                               }
        self.slurm_settings.update(slurm_settings)
        
        
        self.slurm_preamble = []
        self.slurm_preamble.append('deactivate || true')
        self.slurm_preamble.append('conda deactivate || true')
        self.slurm_preamble.append('module purge')


        self.slurm_modules = ['spack_25Q2', '/mnt/projects/sne/Gohar/software_deck/gromacs2025/lib/modulefile']
        self.slurm_commands = []
        self.slurm_commands.append(f"srun -n $SLURM_NTASKS gmx_mpi mdrun -s {self.input_filename}")
        
        
    def uses_plumed(self, plumed_script_path='plumed.dat'):
        self.slurm_commands[-1] = self.slurm_commands[-1].replace("gmx_mpi mdrun", f"gmx_mpi mdrun -plumed {plumed_script_path}")


    def set_enhanced_sampler(self, enhanced_sampler):
        self.enhanced_sampler = enhanced_sampler
    
    def run(self, dry_run=False):
        
        slurm_commands = self.slurm_commands.copy()
        self.slurm_commands = []
        self.slurm_commands.append(f"cd {self.run_dir}")
        self.slurm_commands.extend(self.slurm_precommands)
        self.slurm_commands.extend(slurm_commands)
        self.slurm_commands.extend(self.slurm_postcommands)
        super().run(dry_run=dry_run)
        
        
    def get_completed_nsteps(self):
        raise NotImplementedError("This function is not implemented yet for GROMACS driver.")       
            
            
        