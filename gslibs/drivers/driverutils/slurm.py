"""
SLURM job submission and management utilities.

This module provides functionality for creating, submitting, and monitoring
SLURM jobs for computational chemistry and materials science simulations.
"""

import os
import subprocess

VALID_SBATCH_KEYS = [
    "job-name",
    "partition",
    "time",
    "mem",
    "nodes",
    "ntasks",
    "cpus-per-task",
    "output",
    "error",
    "gres"
]

class slurm_driver:
    """
    SLURM job driver for submitting and managing computational jobs.

    Provides functionality to configure SLURM settings, add modules and commands,
    submit jobs, and monitor job status.

    Parameters
    ----------
    job_name : str
        Name for the SLURM job.
    run_dir : str
        Absolute path to the directory where the job will execute.
    slurm_settings : dict, optional
        Dictionary of SLURM configuration options. Default is empty dict.

    Attributes
    ----------
    job_name : str
        Name of the SLURM job.
    run_dir : str
        Job execution directory.
    job_id : str or None
        SLURM job ID after submission.
    slurm_config : dict
        SLURM configuration parameters.

    Raises
    ------
    ValueError
        If run_dir is not an absolute path.
    """
    def __init__(self, job_name, run_dir, slurm_settings: dict = {}):
        """
        Initialize SLURM driver.

        Parameters
        ----------
        job_name : str
            Name for the SLURM job.
        run_dir : str
            Absolute path to the job execution directory.
        slurm_settings : dict, optional
            Custom SLURM settings to override defaults.
        """
        self.job_name = job_name
        if not run_dir.startswith("/"):
            raise ValueError(f"SLURM: Run directory must be an absolute path.")
        self.run_dir = run_dir
        self.slurm_filename = 'slurm.sh'
        self.slurm_config = {}
        self.slurm_config['job-name'] = job_name
        self.slurm_config['output'] = 'slurm.out'
        self.slurm_config['error'] = 'slurm.err'
        self.slurm_config['nodes'] = 1
        self.slurm_config['ntasks'] = 1
        self.slurm_config['cpus-per-task'] = 8
        self.slurm_config['time'] = '10:00:00'
        self.slurm_config['partition'] = 'zencloud'
        self.slurm_config['mem'] = '20G'

        self.update_config(slurm_settings)

        self.job_id = None
        self.preamble = []
        self.modules = []
        self.commands = []
        self.env_vars = {}
        
    
    def add_preamble(self, line):
        self.preamble.append(line)
    
    def add_module(self, module):
        self.modules.append(module)
    
    def add_command(self, command):
        self.commands.append(command)
    
    def set_partition(self, partition):
        self.slurm_config['partition'] = partition
    
    def set_time(self, time):
        self.slurm_config['time'] = time
    
    def set_mem(self, mem):
        self.slurm_config['mem'] = mem
    
    def add_config(self, key, value):
        if key not in VALID_SBATCH_KEYS:
            raise ValueError(f"SLURM: Invalid SLURM setting key '{key}' provided. Compatible keys are {VALID_SBATCH_KEYS}")
        self.slurm_config[key] = value
    
    def update_config(self, config_dict: dict):
        for k, v in config_dict.items():
            self.add_config(k, v)
    
    def write_slurm(self):
        if not os.path.exists(self.run_dir):
            raise FileNotFoundError(f"SLURM: Directory {self.run_dir} does not exist.")
        slurm_filename = os.path.join(self.run_dir, self.slurm_filename)
        with open(slurm_filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write('\n')
            for key, value in self.slurm_config.items():
                f.write(f"#SBATCH --{key}={value}\n")
            f.write('\n')
            
            for line in self.preamble:
                f.write(f"{line}\n")
            f.write('\n')
            
            for module in self.modules:
                f.write(f"module load {module}\n")
            f.write('\n')
            
            if "OMP_NUM_THREADS" not in self.env_vars:
                self.env_vars["OMP_NUM_THREADS"] = self.slurm_config["cpus-per-task"]
            for key, value in self.env_vars.items():
                f.write(f"export {key}={value}\n")
            f.write('\n')
            
            for command in self.commands:
                f.write(f"{command}\n")
    
    def add_env_var(self, key, value):
        self.env_vars[key] = value
    
    def submit_slurm(self):
        """
        Submit the SLURM job.

        Writes the SLURM script and submits it using sbatch command.

        Returns
        -------
        str
            SLURM job ID.

        Raises
        ------
        ValueError
            If SLURM configuration is incomplete.
        RuntimeError
            If job submission fails.
        """
        if None in self.slurm_config.values():
            raise ValueError(f"SLURM: SLURM configuration not complete. \
                Missing values are {[x for x in self.slurm_config.keys() if self.slurm_config[x] is None]}")
        self.write_slurm()
        
        proc = subprocess.Popen(["sbatch", "slurm.sh"], stdout=subprocess.PIPE, cwd=self.run_dir)
        (out, err) = proc.communicate()
        if err is not None:
            raise RuntimeError(f"SLURM: Error submitting job {self.job_name} to SLURM.")
        self.job_id = out.decode().split(" ")[-1]
        if self.job_id.endswith("\n"):
            self.job_id = self.job_id[:-1]
        return self.job_id
    
    def check_status(self):
        """
        Check the current status of the SLURM job.

        Returns
        -------
        str
            Current job state (e.g., PENDING, RUNNING, COMPLETED, FAILED).

        Raises
        ------
        ValueError
            If job ID is not set or status check fails.
        RuntimeError
            If scontrol command fails.
        """
        if self.job_id is None:
            raise ValueError("SLURM: Job ID not set. Cannot check status.")
        proc = subprocess.Popen(["scontrol", "show", "job", "-o", self.job_id], stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        if err is not None:
            raise RuntimeError(f"SLURM: Error checking status of job {self.job_id}.")
        out = out.decode().split()
        info = {}
        for item in out:
            splititem = item.split("=")
            if len(splititem) > 2:
                key = splititem[0]
                value = "=".join(splititem[1:])
            else:
                key, value = item.split("=")
            info[key] = value
        if 'JobState' not in info:
            raise ValueError(f"SLURM: Error checking status of job {self.job_id}.")
        return info['JobState']
        
    def cancel_job(self):
        if self.job_id is None:
            raise ValueError("SLURM: Job ID not set. Cannot cancel job.")
        proc = subprocess.Popen(["scancel", self.job_id], stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        if err is not None:
            raise RuntimeError(f"SLURM: Error cancelling job {self.job_id}.")
        return out.decode().strip()
        
    
    
        