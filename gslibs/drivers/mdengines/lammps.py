"""
LAMMPS molecular dynamics driver for classical simulations.

This module provides a driver class for running LAMMPS molecular dynamics
simulations through the SLURM job scheduler with enhanced sampling support.
"""

import os
import numpy as np

from gslibs.utils.plumed import *
from gslibs.utils.coordinates_io import *

from gslibs.drivers.base import base_driver

class lammps_driver(base_driver):
    """
    Driver for LAMMPS molecular dynamics simulations.

    Extends base_driver to provide functionality for running LAMMPS MD
    simulations via SLURM job scheduler with support for enhanced sampling
    methods through PLUMED.

    Parameters
    ----------
    input_file : str
        Path to the LAMMPS input file.
    run_name : str
        Name identifier for the run.
    run_dir : str
        Directory where the simulation will be executed.
    verbose : bool, optional
        Enable verbose logging. Default is False.
    meta : dict, optional
        Metadata dictionary. Default is empty dict.
    slurm_settings : dict, optional
        Custom SLURM settings. Default is empty dict.
    """
    def __init__(self, input_file: str,
                 run_name: str,
                 run_dir: str,
                 verbose=False,
                 meta: dict = {},
                 slurm_settings: dict = {},
                 ):
        """
        Initialize LAMMPS driver.

        Parameters
        ----------
        input_file : str
            Path to the LAMMPS input file.
        run_name : str
            Name identifier for the run.
        run_dir : str
            Directory where the simulation will be executed.
        verbose : bool, optional
            Enable verbose logging. Default is False.
        meta : dict, optional
            Metadata dictionary. Default is empty dict.
        slurm_settings : dict, optional
            Custom SLURM settings. Default is empty dict.
        """
        super().__init__(input_file, run_name, run_dir, verbose, meta)

        self.slurm_settings = {"partition": "zencloud", 
                               "time": "3:00:00", 
                               "mem": "20G",
                               "ntasks": "1",
                               "cpus-per-task": "8",
                               }
        self.slurm_settings.update(slurm_settings)
        
        
        self.slurm_preamble = []
        self.slurm_preamble.append('deactivate || true')
        self.slurm_preamble.append('conda deactivate || true')
        self.slurm_preamble.append('module purge')


        self.slurm_modules = ['spack_25Q2', '/mnt/projects/sne/Gohar/software_deck/lammps-ada/lib/modulefile']
        # self.slurm_commands = ['source $SLURM_PRESCRIPT_LAMMPS']
        self.slurm_commands.append(f"srun -n $SLURM_NTASKS lmp -i {self.input_filename}")
        
        self.enhanced_sampler = None
        
    def is_fail_frame_criterion_reached(self, method, save_pdb, **kwargs):
        if not os.path.exists(f"{self.run_dir}/lammps.traj"):
            return False
        
        if method == "COLVAR":
            colvar_file = kwargs.get("colvar_file", "COLVAR.fixed")
            if not os.path.exists(f"{self.run_dir}/{colvar_file}"):
                return False

            kwargs = kwargs if kwargs is not None else {}
            metric = kwargs.get("metric", "CV")
            criterion_type = kwargs.get("criterion_type", 'greater')
            threshold = kwargs.get("threshold", 0.5)
            
            metric_values = plumed_outfile_reader(f"{self.run_dir}/{colvar_file}", columns=[metric])
            
            traj = self.read_trajectory_file(f"{self.run_dir}/lammps.traj", use_meta=True)
            
            if save_pdb:
                pdbtraj = ase_to_pdb(traj)
                with open(f"{self.run_dir}/traj.pdb", 'w') as f:
                    f.write(pdbtraj)
                self.log_line(f"Created pdb trajectory at {self.run_dir}/traj.pdb")
            
            if len(metric_values) != len(traj):
                traj = traj[:len(metric_values)]
            
            if criterion_type == 'greater':
                reject_criterion = lambda x: x > threshold
            elif criterion_type == 'lesser':
                reject_criterion = lambda x: x < threshold
            else:
                raise ValueError(f"LAMMPS {self.run_name}: Criterion type {criterion_type} not recognized.")
            
            metric_values = metric_values[metric].to_numpy()
            fail_index = np.argmax(reject_criterion(metric_values))
            
            # print(f"\n\n {reject_criterion(metric_values[fail_index])} \n\n")
            
            return reject_criterion(metric_values[fail_index])
        
            
    def create_fail_frame(self, method, save_pdb=False, **kwargs):
        """
        Extract failed frame from simulation based on specified criteria.

        Parameters
        ----------
        method : str
            Method for identifying fail frame. Currently supports "COLVAR".
        save_pdb : bool, optional
            Whether to save trajectory as PDB file. Default is False.
        **kwargs
            Additional arguments for fail frame detection:
            - colvar_file : str, optional
                COLVAR file name. Default is "COLVAR.fixed".
            - metric : str, optional
                Metric column name. Default is "CV".
            - criterion_type : str, optional
                Comparison type ('greater' or 'lesser'). Default is 'greater'.
            - threshold : float, optional
                Threshold value. Default is 0.5.

        Raises
        ------
        ValueError
            If run status is not completed or method not recognized.
        FileNotFoundError
            If trajectory or COLVAR file not found.
        """
        if self.status not in ["COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"]:
            raise ValueError(f"LAMMPS {self.run_name}: Run status is {self.status} not COMPLETED, CANCELLED or FAILED. Cannot create fail frame.")
        if not os.path.exists(f"{self.run_dir}/lammps.traj"):
            raise FileNotFoundError(f"LAMMPS {self.run_name}: Completed LAMMPS trajectory file not found.")
        
        if method == "COLVAR":
            self.log_line("Using COLVAR method to create fail frame.")
            colvar_file = kwargs.get("colvar_file", "COLVAR.fixed")
            if not os.path.exists(f"{self.run_dir}/{colvar_file}"):
                raise FileNotFoundError(f"LAMMPS {self.run_name}: Tried to create fail frame by COLVAR method but colvar file {self.run_dir}/{colvar_file} not found.")

            kwargs = kwargs if kwargs is not None else {}
            metric = kwargs.get("metric", "CV")
            criterion_type = kwargs.get("criterion_type", 'greater')
            threshold = kwargs.get("threshold", 0.5)
            
            self.log_line(f"Checking file {colvar_file} using metric {metric} with criterion type {criterion_type} and threshold {threshold}.")
            
            metric_values = plumed_outfile_reader(f"{self.run_dir}/{colvar_file}", columns=[metric])
            
            traj = self.read_trajectory_file(f"{self.run_dir}/lammps.traj", use_meta=True)
            
            if save_pdb:
                pdbtraj = ase_to_pdb(traj)
                with open(f"{self.run_dir}/traj.pdb", 'w') as f:
                    f.write(pdbtraj)
                self.log_line(f"Created pdb trajectory at {self.run_dir}/traj.pdb")
            
            if len(metric_values) != len(traj):
                self.log_line(f"Number of frames in the trajectory file and the COLVAR file do not match. Will trim the trajectory file.")
                traj = traj[:len(metric_values)]
            
            if criterion_type == 'greater':
                reject_criterion = lambda x: x > threshold
            elif criterion_type == 'lesser':
                reject_criterion = lambda x: x < threshold
            else:
                raise ValueError(f"LAMMPS {self.run_name}: Criterion type {criterion_type} not recognized.")
            
            metric_values = metric_values[metric].to_numpy()
            fail_index = np.where(reject_criterion(metric_values) == True)[0]
            fail_index = fail_index[0] if len(fail_index) > 0 else -1 
            
            if fail_index == -1:
                self.log_line("No frame failed the accept criterion. Taking last frame.")
            else:
                self.log_line(f"Frame number {fail_index + 1} is the fail frame.")
                
            fail_frame = traj[fail_index]
            fail_xyz = ase_to_xyz(fail_frame)
            
            with open(f"{self.run_dir}/fail_frame.xyz", 'w') as f:
                f.write(fail_xyz)
            self.log_line(f"Created fail frame at {self.run_dir}/fail_frame.xyz")
        else:
            raise ValueError(f"LAMMPS {self.run_name}: Method {method} not recognized.")
    
    def set_enhanced_sampler(self, enhanced_sampler):
        self.enhanced_sampler = enhanced_sampler
    
    def run(self, dry_run=False):
        if self.enhanced_sampler is None:
            if "plumed.dat" in self.aux_files.keys():
                self.template_aux_file(dest="plumed.dat", template={"${LOAD_SECTION}": ''})
                self.template_aux_file(dest="plumed.dat", template={"${METAD_SECTION}": ''})
        else:
            plumed_custom_keywords = self.enhanced_sampler.get_plumed_custom_keywords(self.run_dir)
            templ = []
            for fi in plumed_custom_keywords:
                filename = os.path.basename(fi)
                self.add_aux_files(source=fi, dest=filename)
                templ.append(f"LOAD FILE={filename}")
            templ = "\n".join(templ)
            self.template_aux_file(dest="plumed.dat", 
                                   template={"${LOAD_SECTION}": templ})
            self.template_aux_file(dest="plumed.dat", 
                                   template={"${METAD_SECTION}": self.enhanced_sampler.get_plumed_section()})
            if self.enhanced_sampler.requires_training():
                self.add_binary_file(source=self.enhanced_sampler.get_encoder_path(), 
                                   dest="encoder.pt")
        
        slurm_commands = self.slurm_commands.copy()
        self.slurm_commands = []
        self.slurm_commands.append(f"cd {self.run_dir}")
        self.slurm_commands.extend(self.slurm_precommands)
        self.slurm_commands.extend(slurm_commands)
        self.slurm_commands.extend(self.slurm_postcommands)
        
        super().run(dry_run=dry_run)
        
        
    def get_fail_frame_path(self):
        ffpath = f"{self.run_dir}/fail_frame.xyz"
        if not os.path.exists(ffpath):
            raise FileNotFoundError(f"LAMMPS {self.run_name}: Fail frame not found at {ffpath}")
        return ffpath
    
    def get_completed_nsteps(self):
        if not os.path.exists(f"{self.run_dir}/lammps.traj"):
            return 0
        traj = self.read_trajectory_file(f"{self.run_dir}/lammps.traj", use_meta=True)
        return len(traj)
            
            
            
        