import os
from os.path import exists, isdir
import warnings
import shutil
import yaml
from ase.data import atomic_numbers

from gsutils.drivers.driverutils.slurm import slurm_driver
from gsutils.utils.io_coordinates import *

from gsutils.base import base_driver


SERIALIZED_MODEL_FILENAME = "mlff_deployed.pth"

class mace_driver(base_driver):
    def __init__(self, input_file, run_name, run_dir, verbose=False, meta={}):
        super().__init__(input_file, run_name, run_dir, verbose, meta)
        assert self.input_file.endswith(".yaml"), f"base_config {self.input_file} must be in .yaml format"
        
        self.input_filename = 'base_config.yaml'
        
        self.config = yaml.load("".join(self.input_script), Loader=yaml.FullLoader)
        
        self.config['name'] = run_name
        self.config['wandb_name'] = run_name
        self.config['train_file'] = 'train_data.xyz'
        self.config['work_dir'] = run_dir
        self.config['device'] = 'cuda'

        if 'specorder' in meta.keys():
            atn_list = []
            for ind,value in enumerate(meta['specorder']):
                atn_list.append(atomic_numbers[value])
        
        self.config['atomic_numbers'] = f"[{",".join([str(x) for x in atn_list])}]"
        
        self.train_data = []
        self.eval_data = []
        self.slurm_settings = {"partition": "gpucloud", 
                               "time": "12:00:00", 
                               "mem": "24G",
                               "cpus-per-task": "8",
                               "gpus": "1",}
        file_dir = os.path.dirname(os.path.realpath(__file__))
        venv_activate = os.path.join(file_dir, "../../.venv/bin/activate")
        venv_activate = os.path.abspath(venv_activate)
        self.slurm_preamble.append("source {}".format(venv_activate))
        self.slurm_commands.append("srun mace_run_train --config=\"run_config.yaml\" || exit 1")
        # self.slurm_commands.append(f"MACE-deploy build --train-dir ./{run_name} {SERIALIZED_MODEL_FILENAME} || exit 1")
        
        
        self.evaluation_dir = None
        self.evaluate_commands = []
        self.evaluate_commands.append("MACE-evaluate --train-dir ./train_dir --dataset-config eval_config.yaml --metrics-config None --output eval_output.xyz")
        
    def additional_load_from_module(self, module):
        self.train_data = module.train_data
    
    def config_update(self, entries: dict):
        self.config.update(entries)
        
    def get_train_dataset(self):
        if self.check_status() != "COMPLETED":
            raise ValueError(f"MACE {self.run_name}: Model not yet completed. Current status: {self.status}")
        
        train_data_file = os.path.join(self.run_dir, "train_data.xyz")
        return train_data_file
    
    def add_training_data(self, frame_file):
        assert exists(frame_file), f"frame_file {frame_file} does not exist. Please provide a valid path (use absolute path)."
        assert frame_file.endswith(".xyz"), f"frame_file {frame_file} must be in .xyz format"
        
        self.train_data.append(frame_file)
    
    def add_eval_data(self, frame_file):
        assert exists(frame_file), f"frame_file {frame_file} does not exist. Please provide a valid path (use absolute path)."
        assert frame_file.endswith(".xyz"), f"frame_file {frame_file} must be in .xyz format"
        
        self.eval_data.append(frame_file)
    
    def write_files(self):
        super().write_files()
        with open(f"{self.run_dir}/run_config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        self.log_line(f"Wrote run config to {self.run_dir}/run_config.yaml")
        
        with open(f"{self.run_dir}/train_data.xyz", 'w') as dataf:
            for frame_file in self.train_data:
                with open(frame_file, 'r') as framef:
                    read_data = framef.read()
                    dataf.write(read_data)
        self.log_line(f"Wrote training data to {self.run_dir}/train_data.xyz")
    
    
    def write_evaluation_files(self):
        with open(f"{self.run_dir}/{self.input_filename}", 'w') as f:
            f.writelines(self.input_script)
        self.log_line(f"Wrote run config to {self.run_dir}/eval_config.yaml")
        
        for dest, data in self.aux_files.items():
            with open(f"{self.run_dir}/{dest}", 'w') as f:
                f.writelines(data)
            self.log_line(f"Wrote file {dest} to {self.run_dir}")
        
        with open(f"{self.run_dir}/eval_data.xyz", 'w') as dataf:
            for frame_file in self.eval_data:
                with open(frame_file, 'r') as framef:
                    read_data = framef.read()
                    dataf.write(read_data)
        self.log_line(f"Wrote evaluation data to {self.run_dir}/eval_data.xyz")
        
        shutil.copytree(self.evaluation_dir, 
                        os.path.join(self.run_dir, "train_dir"))
    
    
    def train(self, dry_run=False):
        self.run(dry_run)

    def get_serialized_model_file(self, complete_check=True):
        if complete_check:
            if self.status != "COMPLETED":
                raise ValueError(f"MACE {self.run_name}: Model not yet completed. Current status: {self.status}")
        return os.path.join(self.run_dir, SERIALIZED_MODEL_FILENAME)
    
    def get_train_data_file(self):
        train_data = os.path.join(self.run_dir, "train_data.xyz")
        if not exists(train_data):
            raise ValueError(f"MACE {self.run_name}: Training data {train_data} does not exist.")
        return train_data
    
    def set_evaluation_dir(self, eval_dir):
        self.evaluation_dir = eval_dir
    
    def evaluate(self, dry_run=False):
        if self.evaluation_dir is None:
            raise ValueError(f"MACE {self.run_name}: Evaluation directory not set.")
        # dry_run = True
        if len(self.slurm_commands) == 0:
            raise ValueError(self.get_log_lines(f"No commands added to run."))
        self.create_run_dir()
        self.write_evaluation_files()

        self.slurm_job = slurm_driver(self.run_name, 
                                      self.run_dir,
                                      partition=self.slurm_settings['partition'],
                                      time=self.slurm_settings['time'],
                                    mem=self.slurm_settings['mem'])
        remaining_keys = [key for key in self.slurm_settings.keys() if key not in ["partition", "time", "mem"]]
        for key in remaining_keys:
            self.slurm_job.add_config(key, self.slurm_settings[key])
            
        for module in self.slurm_modules:
            self.slurm_job.add_module(module)
            
        for line in self.slurm_preamble:
            self.slurm_job.add_preamble(line)

        for key, value in self.runtime_env.items():
            self.slurm_job.add_env_var(key, value)
        
        for line in self.evaluate_commands:
            self.slurm_job.add_command(line)
        
        self.slurm_job.write_slurm()
        if dry_run:
            return
        self.slurm_job.submit_slurm()
        self.log_line(f"Submitted job with job id {self.slurm_job.job_id}")

        
        
        