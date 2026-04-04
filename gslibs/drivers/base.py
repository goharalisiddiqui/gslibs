import os
import numpy as np
from ase.data import atomic_numbers
from ase import Atoms
from typing import Callable
from time import sleep

from gslibs.drivers.driverutils.slurm import slurm_driver
from gslibs.utils.coordinates_io import *

class base_driver:
    """
    Base driver class for computational chemistry and materials science simulations.

    This class provides a common interface for running computational simulations
    using SLURM job scheduler. It handles file management, coordinate conversion,
    and job execution for various computational engines.

    Parameters
    ----------
    input_file : str
        Absolute path to the input file for the simulation.
    run_name : str
        Name identifier for the simulation run.
    run_dir : str
        Absolute path to the directory where the simulation will be executed.
    verbose : bool, optional
        Whether to enable verbose logging. Default is False.
    meta : dict, optional
        Dictionary containing metadata such as cell parameters, species order, etc.
    slurm_settings : dict, optional
        Dictionary containing SLURM job settings. Default settings are provided.

    Attributes
    ----------
    input_file : str
        Path to the input file.
    run_name : str
        Name of the simulation run.
    run_dir : str
        Directory where simulation files are stored.
    verbose : bool
        Logging verbosity flag.
    meta : dict
        Metadata dictionary.
    status : str
        Current status of the simulation job.
    coordinate_files : dict
        Dictionary of coordinate files to be written.
    aux_files : dict
        Dictionary of auxiliary files to be written.
    binary_files : dict
        Dictionary of binary files to be written.
    copy_directories : dict
        Dictionary of directories to be copied.
    slurm_job : slurm_driver or None
        SLURM job driver instance.
    slurm_settings : dict
        SLURM job configuration settings.

    Raises
    ------
    AssertionError
        If required parameters are None or empty.
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If paths are not absolute or run_dir already exists.
    """
    def __init__(self, input_file: str,
                 run_name: str,
                 run_dir: str,
                 verbose: bool = False,
                 meta : dict = {},
                 slurm_settings: dict = {},
                 ):
        """
        Initialize the base driver.

        Parameters
        ----------
        input_file : str
            Absolute path to the input file for the simulation.
        run_name : str
            Name identifier for the simulation run.
        run_dir : str
            Absolute path to the directory where the simulation will be executed.
        verbose : bool, optional
            Whether to enable verbose logging. Default is False.
        meta : dict, optional
            Dictionary containing metadata. Default is empty dict.
        slurm_settings : dict, optional
            Dictionary containing SLURM job settings. Default is empty dict.
        """
        assert None not in [input_file, run_name, run_dir], "input_file, run_name and run_dir must be provided"
        assert 0 not in [len(input_file), len(run_name), len(run_dir)], "input_file, run_name and run_dir must be non-empty"
        assert input_file.startswith('/'), "input_file must be an absolute path"
        assert os.path.exists(input_file), f"File {input_file} not found."
        assert run_dir.startswith('/'), "run_dir must be an absolute path"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"{type(self).__name__} : File {input_file} not found.")
        
        # if os.path.exists(run_dir):
        #     raise ValueError(f"run_dir {run_dir} already exists")
        
        self.input_file = input_file
        self.input_filename = os.path.basename(input_file)
        try:
            with open(input_file, 'r') as f:
                input_script = f.readlines()
            self.input_is_binary = False
            self.input_script = input_script
        except UnicodeDecodeError:
            self.input_is_binary = True
        
        self.run_name = str(run_name)
        self.verbose = verbose
        self.meta = meta
        
        self.log_line(f"Initialized {type(self).__name__} driver code {run_name} with input file {input_file}")
        self.run_dir = run_dir
        self.coordinate_files = {}
        self.aux_files = {}
        self.binary_files = {}
        self.copy_directories = {}
        self.slurm_job = None
        self.status = "INITIALIZED"
        
        # Settings for child classes
        self.slurm_settings = {"partition": "zencloud", 
                               "time": "10:00:00", 
                               "mem": "20G"}
        self.slurm_settings.update(slurm_settings)
        self.runtime_env = {}
        self.slurm_preamble = []
        self.slurm_modules = []
        self.slurm_precommands = []
        self.slurm_commands = []
        self.slurm_postcommands = []
    
    def load_from_module(self, module):
        """
        Load driver state from another module instance.

        Parameters
        ----------
        module : object
            Module instance to load state from.

        Raises
        ------
        ValueError
            If module is None.
        """
        if module is None:
            raise ValueError(self.get_log_lines(f"Module is None."))
        self.slurm_job = module.slurm_job
        self.status = module.status
        self.additional_load_from_module(module)
        self.log_line(f"Loaded from {module}")
    
    def additional_load_from_module(self, module):
        """
        Load additional variables from module (to be overridden in child classes).

        Parameters
        ----------
        module : object
            Module instance to load additional variables from.
        """
        # Overwrite this function in the child class to add any additional variables to load from module
        pass
    
    def log_line(self, line):
        """
        Log a message with entity name prefix if verbose mode is enabled.

        Parameters
        ----------
        line : str
            Message to log.
        """
        entity_name = type(self).__name__
        entity_name = entity_name.split("_")[0].upper()
        print(f"{entity_name} {self.run_name}: {line}") if self.verbose else None
    
    def get_log_lines(self, line):
        """
        Generate log line with entity name prefix.

        Parameters
        ----------
        line : str
            Message to format.

        Returns
        -------
        str
            Formatted log line with entity prefix.
        """
        entity_name = type(self).__name__
        entity_name = entity_name.split("_")[0].upper()
        return f"{entity_name} {self.run_name}: {line}"
    
    def add_runtime_env(self, key, value):
        """
        Add runtime environment variable for the job.

        Parameters
        ----------
        key : str
            Environment variable name.
        value : str
            Environment variable value.
        """
        self.runtime_env[key] = value
        self.log_line(f"Added runtime environment variable {key} with value {value}")
        
    
    def read_trajectory_file(self, source: str, use_meta=False):
        """
        Read trajectory file and convert to ASE atoms objects.

        Parameters
        ----------
        source : str
            Path to the trajectory file.
        use_meta : bool, optional
            Whether to use metadata for reading. Default is False.

        Returns
        -------
        list of ase.Atoms or ase.Atoms
            Trajectory data as ASE atoms objects.

        Raises
        ------
        FileNotFoundError
            If the source file does not exist.
        ValueError
            If file format is unsupported or required metadata is missing.
        """
        if not os.path.exists(source):
            raise FileNotFoundError(self.get_log_lines(f"File {source} not found."))
        if source.endswith(".lammps"):
            data = lammpsdata_to_ase(source)
        elif source.endswith(".traj"):
            data = lammpsdump_to_ase(source)
            if use_meta:
                if not "specorder" in self.meta.keys():
                    raise ValueError(self.get_log_lines(f"Specorder not found in metadata. Required to read LAMMPS trajectory files."))
                right_atns = [atomic_numbers[elem] for elem in self.meta["specorder"]]
                for frame in data:
                    wrong_atns = frame.get_atomic_numbers()
                    correct_atns = [right_atns[wrong_atn - 1] for wrong_atn in wrong_atns]
                    frame.set_atomic_numbers(correct_atns)
        elif source.endswith(".pdb"):
            if use_meta:
                if not "cell" in self.meta.keys():
                    raise ValueError(self.get_log_lines(f"Cell information not found in metadata. Required to read PDB files."))
                data = pdb_to_ase(source, cell=self.meta["cell"], pbc=self.meta["pbc"] if "pbc" in self.meta.keys() else None)
            else:
                data = pdb_to_ase(source)
                
        elif source.endswith(".xyz"):
            if use_meta:
                if not "cell" in self.meta.keys():
                    raise ValueError(self.get_log_lines(f"Cell information not found in metadata. Required to read PDB files."))
                data = xyz_to_ase(source, cell=self.meta["cell"], pbc=self.meta["pbc"] if "pbc" in self.meta.keys() else None)
            else:
                data = xyz_to_ase(source)
        else:
            raise ValueError(self.get_log_lines(f"Unsupported file type for {source}"))
        return data

    def prepare_coordinates_file(self, source: Atoms, dest: str, use_meta=False, extxyz=True):
        """
        Prepare coordinate file content from ASE atoms object.

        Parameters
        ----------
        source : ase.Atoms
            ASE atoms object containing the structure.
        dest : str
            Destination file path to determine output format.
        use_meta : bool, optional
            Whether to use metadata for conversion. Default is False.
        extxyz : bool, optional
            Whether to use extended XYZ format. Default is True.

        Returns
        -------
        str
            Coordinate file content as string.

        Raises
        ------
        ValueError
            If source is not an ASE Atoms object, cell information is missing,
            or file format is unsupported.
        """
        if not isinstance(source, Atoms):
            raise ValueError(self.get_log_lines(f"Source must be an ASE Atoms object."))
        cell = source.get_cell()
        if (cell == 0).all():
            if use_meta:
                if "cell" not in self.meta.keys():
                    raise ValueError(self.get_log_lines(f"Cell information not found in metadata. Required to use_meta to write ASE object without cell to coordinates."))
                cell = self.meta["cell"]
            else:
                raise ValueError(self.get_log_lines(f"Cell information not found in ASE object. Use use_meta=True to use cell information from metadata."))

        if dest.endswith(".lammps"):
            if not use_meta:
                raise ValueError(self.get_log_lines(f"Specorder required to write LAMMPS data file. Use use_meta=True to use specorder from metadata."))
            if "specorder" not in self.meta.keys():
                raise ValueError(self.get_log_lines(f"Specorder not found in metadata. Cannot write LAMMPS data file without."))
            extra_info = {"specorder": self.meta["specorder"]}
            if "lammps_atom_style" in self.meta.keys():
                extra_info["atom_style"] = self.meta["lammps_atom_style"]
            if "lammps_units" in self.meta.keys():
                extra_info["units"] = self.meta["lammps_units"]
            output = ase_to_lammpsdata(source, extra_info)
        elif dest.endswith(".pdb"):
            output = ase_to_pdb(source)
        elif dest.endswith(".xyz"):
            output = ase_to_xyz(source, extxyz=extxyz)
        else:
            raise ValueError(self.get_log_lines(f"Unsupported file type for {dest}"))
        return output
    
    def add_coordinates_file(self, source, dest: str, perturb=None, extxyz=True):
        """
        Add coordinate file to be written during job execution.

        Parameters
        ----------
        source : str or ase.Atoms
            Source coordinate file path or ASE atoms object.
        dest : str
            Destination filename within the run directory.
        perturb : float, optional
            Random perturbation amplitude in Angstroms. Default is None.
        extxyz : bool, optional
            Whether to use extended XYZ format for XYZ files. Default is True.
        """
        dest = str(dest)
        
        if isinstance(source, Atoms):
            data = [source]
        else:
            data = self.read_trajectory_file(source, use_meta=True)
        
        if len(data) > 1:
            print(f"{type(self).__name__} {self.run_name}: Warning: Multiple frames found in coordinates file {source}. Using the first frame.")
        data = data[0]
        
        if perturb is not None:
            data.set_positions(data.get_positions() + np.random.uniform(-perturb, perturb, data.get_positions().shape))
        
        output = self.prepare_coordinates_file(data, dest, use_meta=True, extxyz=extxyz)
        self.coordinate_files[dest] = output.split("\n")
        
        self.log_line(f"Added coordinates file {source} as {dest}")

    def add_aux_files(self, source, dest):
        self.add_aux_file(source, dest) # For backward compatibility

    def add_aux_file(self, source, dest):
        """
        Add auxiliary file to be copied during job execution.

        Parameters
        ----------
        source : str
            Source file path.
        dest : str
            Destination filename within the run directory.

        Raises
        ------
        FileNotFoundError
            If the source file does not exist.
        AssertionError
            If destination filename is empty.
        """
        if not os.path.exists(source):
            raise FileNotFoundError(self.get_log_lines(f"File {source} not found."))
        assert len(dest) > 0, f"{type(self).__name__} {self.run_name}: Destination file must be non-empty."
        dest = str(dest)
        
        with open(source, 'r') as f:
            data = f.readlines()
        self.aux_files[dest] = data
        self.log_line(f"Added auxiliary file {source} as {dest}")
    
    def add_binary_file(self, source, dest):
        """
        Add binary file to be copied during job execution.

        Parameters
        ----------
        source : str
            Source binary file path.
        dest : str
            Destination filename within the run directory.

        Raises
        ------
        FileNotFoundError
            If the source file does not exist.
        AssertionError
            If destination filename is empty.
        """
        if not os.path.exists(source):
            raise FileNotFoundError(self.get_log_lines(f"File {source} not found."))
        assert len(dest) > 0, f"{type(self).__name__} {self.run_name}: Destination file must be non-empty."
        dest = str(dest)

        with open(source, 'rb') as f:
            data = f.read()
        self.binary_files[dest] = data
        self.log_line(f"Added binary file {source} as {dest}")

    def add_copy_directory(self, source, dest):
        if not os.path.exists(source):
            raise FileNotFoundError(self.get_log_lines(f"Directory {source} not found."))
        assert len(dest) > 0, f"{type(self).__name__} {self.run_name}: Destination directory must be non-empty."
        dest = str(dest)

        self.copy_directories[dest] = source
        self.log_line(f"Added copy directory {source} as {dest}")
        
    def template_aux_file(self, dest, template):
        """
        Apply template substitutions to auxiliary or input file.

        Parameters
        ----------
        dest : str
            Destination file name to template.
        template : dict
            Dictionary of string replacements {old: new}.

        Raises
        ------
        ValueError
            If file not found, is binary, or template keywords not found.
        """
        if dest is not self.input_filename and dest not in self.aux_files.keys():
            raise ValueError(self.get_log_lines(f"File {dest} not found to template."))
        found_keywords = [False for _ in range(len(template))]
        if dest == self.input_filename:
            if self.input_is_binary:
                raise ValueError(self.get_log_lines(f"Input file {self.input_filename} is binary. Cannot template binary files."))
            file_to_template = self.input_script
        else:
            file_to_template = self.aux_files[dest]
        for index, old_line in enumerate(file_to_template):
            new_line = old_line
            for ind, (key, value) in enumerate(template.items()):
                if key in old_line:
                    new_line = new_line.replace(key, value)
                    found_keywords[ind] = True
            if dest == self.input_filename:
                self.input_script[index] = new_line
            else:
                self.aux_files[dest][index] = new_line
        template_keywords = list(template.keys())
        not_found = [template_keywords[ind] for ind in range(len(template_keywords)) if not found_keywords[ind]]
        if len(not_found) > 0:
            raise ValueError(self.get_log_lines(f"Could not find keywords {','.join(not_found)} in destination file {dest}."))
        self.log_line(f"Template {dest} with {template}")
    
    def template_input_script(self, template):
        self.template_aux_file(self.input_filename, template)
        
    def write_files(self):
        if not self.status == "CREATED":
            raise ValueError(self.get_log_lines(f"Run not created. Cannot write files."))
        
        if self.input_is_binary:
            with open(f"{self.run_dir}/{self.input_filename}", 'wb') as f:
                with open(self.input_file, 'rb') as fin:
                    f.write(fin.read())
            self.log_line(f"Copied binary input file to {self.run_dir}/{self.input_filename}")
        else:
            with open(f"{self.run_dir}/{self.input_filename}", 'w') as f:
                f.writelines(self.input_script)
            self.log_line(f"Wrote input script to {self.run_dir}/{self.input_filename}")
        
        for dest, data in self.coordinate_files.items():
            with open(f"{self.run_dir}/{dest}", 'w') as f:
                f.writelines([line + "\n" for line in data])
            self.log_line(f"Wrote coordinate file {dest} to {self.run_dir}")
            
        for dest, data in self.aux_files.items():
            with open(f"{self.run_dir}/{dest}", 'w') as f:
                f.writelines([line + "\n" for line in data])
            self.log_line(f"Wrote file {dest} to {self.run_dir}")
        
        for dest, data in self.binary_files.items():
            with open(f"{self.run_dir}/{dest}", 'wb') as f:
                f.write(data)
            self.log_line(f"Wrote binary file {dest} to {self.run_dir}")
        
        for dest, source in self.copy_directories.items():
            os.system(f"cp -r {source} {self.run_dir}/{dest}")
            self.log_line(f"Copied directory {source} to {self.run_dir}/{dest}")
    
    def check_status(self):
        """
        Check the current status of the SLURM job.

        Returns
        -------
        str
            Current job status (PENDING, RUNNING, COMPLETED, etc.).
        """
        if self.status in ["COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"]:
            return self.status
        if self.slurm_job is None:
            return self.status
        self.status = self.slurm_job.check_status()
        return self.status
    
    def is_job_running(self):
        return self.check_status() in ["PENDING", "RUNNING", "COMPLETING"]

    def create_run_dir(self):
        if self.run_dir is None:
            raise ValueError(self.get_log_lines(f"Run directory not set."))
        if os.path.exists(self.run_dir):
            raise ValueError(self.get_log_lines(f"Run directory {self.run_dir} already exists."))
        os.makedirs(self.run_dir)
        self.log_line(f"Created run directory {self.run_dir}")
        self.status = "CREATED"

    def run(self, dry_run=False):
        """
        Execute the simulation job via SLURM.

        Creates run directory, writes files, sets up SLURM job, and submits it.

        Parameters
        ----------
        dry_run : bool, optional
            If True, prepares job but does not submit. Default is False.

        Raises
        ------
        ValueError
            If no commands have been added to run.
        """
        # dry_run = True
        if len(self.slurm_commands) == 0:
            raise ValueError(self.get_log_lines(f"No commands added to run."))
        self.create_run_dir()
        self.write_files()

        self.slurm_job = slurm_driver(self.run_name, 
                                      self.run_dir,
                                      self.slurm_settings)

        for module in self.slurm_modules:
            self.slurm_job.add_module(module)
            
        for line in self.slurm_preamble:
            self.slurm_job.add_preamble(line)

        for key, value in self.runtime_env.items():
            self.slurm_job.add_env_var(key, value)
        
        for command in self.slurm_commands:
            self.slurm_job.add_command(command)
        self.prerun()
        self.slurm_job.write_slurm()
        if dry_run:
            return
        self.slurm_job.submit_slurm()
        self.log_line(f"Submitted job with job id {self.slurm_job.job_id}")
    
    def prerun(self):
        # Overwrite this function in the child class to add any prerun commands
        pass

    def monitor_from_file(self, filename, column_index : int, cancel_criteria: Callable, interval=5):
        """
        Monitor a file and cancel job if criteria are met.

        Parameters
        ----------
        filename : str
            Name of the file to monitor in the run directory.
        column_index : int
            Column index to monitor in the file.
        cancel_criteria : callable
            Function that returns True if job should be cancelled.
        interval : int, optional
            Monitoring interval in seconds. Default is 5.

        Raises
        ------
        ValueError
            If SLURM job is not running.
        """
        elapsed_time = 0
        if not self.is_job_running():
            raise ValueError(self.get_log_lines(f"Slurm job not running. Status is {self.status}"))
        filepath = os.path.join(self.run_dir, filename)
        if not os.path.exists(filepath):
            print(f"{type(self).__name__} {self.run_name}: Waiting for file {filename} to appear...")
            while not os.path.exists(filepath) and self.is_job_running():
                sleep(interval)
                elapsed_time += interval
        elapsed_time = 0
        print(f"{type(self).__name__} {self.run_name}: Monitoring file {filename}...")
        while self.is_job_running():
            with open(filepath, 'r') as f:
                line = f.readlines()
            if len(line) < 2:
                sleep(interval)
                elapsed_time += interval
                continue
            last_line = line[-1]
            if len(last_line.split()) < column_index:
                sleep(interval)
                elapsed_time += interval
                continue
            if cancel_criteria(float(last_line.split()[column_index])):
                self.slurm_job.cancel_job()
                self.log_line(f"job cancelled via monitoring file {filename}.")
                break
    
    def get_check_from_file(self, filename, column_index : int, criterion_type: str, threshold: float):
        if self.check_status() in ["INITIALIZED", "CREATED"]:
            return
        filepath = os.path.join(self.run_dir, filename)
        if not os.path.exists(filepath):
            return
        with open(filepath, 'r') as f:
            line = f.readlines()
        if len(line) < 2:
            return
        last_line = line[-1]
        if len(last_line.split()) < column_index:
            return
        val = float(last_line.split()[column_index])
        if criterion_type == "greater":
            return val > threshold
        elif criterion_type == "lesser":
            return val < threshold
        elif criterion_type == "equal":
            return val == threshold
        else:
            raise ValueError(self.get_log_lines(f"Unsupported criterion type {criterion_type}"))
            
    def check_from_file(self, filename, column_index : int, criterion_type: str, threshold: float):
        if not self.is_job_running():
            return
        result = self.get_check_from_file(filename, column_index, criterion_type, threshold)
        if result is True:
            self.slurm_job.cancel_job()
            self.log_line(f"job cancelled via monitoring file {filename}.")
