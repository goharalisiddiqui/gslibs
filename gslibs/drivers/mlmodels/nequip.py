import os
import shutil
import yaml
from os.path import exists, isdir

from gsutils.drivers.driverutils.slurm import slurm_driver
from gsutils.drivers.base import base_driver


SERIALIZED_MODEL_FILENAME = "model.nequip.zip"

class nequip_driver(base_driver):
    def __init__(self, input_file, run_name, run_dir, verbose=False, meta={}):
        super().__init__(input_file, run_name, run_dir, verbose, meta)
        assert self.input_file.endswith(".yaml"), f"base_config {self.input_file} must be in .yaml format"
        
        self.input_filename = 'base_config.yaml'
        
        
        
        self.config = {}
        self.config['ROOT_DIR'] = run_dir
        self.config['RUN_NAME'] = run_name
        self.config['DATASET'] = 'train_data.xyz'
        if 'specorder' in meta.keys():
            spec_list = []
            for symbol in meta['specorder']:
                spec_list.append(symbol)
            self.config['chemical_symbols'] = spec_list
        
        self.train_data = []
        self.eval_data = []
        self.slurm_settings = {"partition": "gpucloud", 
                               "time": "12:00:00", 
                               "mem": "24G",
                               "gres": "shard:4",}
        self.slurm_modules = ["cuda/12.2"]
        venv_activate = os.path.join(os.getcwd(), ".venv", "bin", "activate")
        venv_activate = os.path.abspath(venv_activate)
        self.slurm_preamble.append("source {}".format(venv_activate))
        self.slurm_commands.append("srun nequip-train --config-name run_config.yaml || exit 1")
        self.slurm_commands.append(f"nequip-package --ckpt-path ./best.ckpt --output-path ./{SERIALIZED_MODEL_FILENAME} || exit 1")
        self.evaluation_dir = None
        
    def additional_load_from_module(self, module):
        self.train_data = module.train_data
    
    def config_update(self, entries: dict):
        self.config.update(entries)
        
    def get_train_dataset(self):
        if self.check_status() != "COMPLETED":
            raise ValueError(f"NEQUIP {self.run_name}: Model not yet completed. Current status: {self.status}")
        
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
        with open(f"{self.run_dir}/run_config.yaml", 'a') as f:
            with open(self.input_file, 'r') as basef:
                read_data = basef.read()
                f.write(read_data)
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
                raise ValueError(f"NEQUIP {self.run_name}: Model not yet completed. Current status: {self.status}")
        return os.path.join(self.run_dir, SERIALIZED_MODEL_FILENAME)
    
    def get_train_data_file(self):
        train_data = os.path.join(self.run_dir, "train_data.xyz")
        if not exists(train_data):
            raise ValueError(f"NEQUIP {self.run_name}: Training data {train_data} does not exist.")
        return train_data
    
    def set_evaluation_dir(self, eval_dir):
        self.evaluation_dir = eval_dir
    
    def evaluate(self, dry_run=False):
        if self.evaluation_dir is None:
            raise ValueError(f"NEQUIP {self.run_name}: Evaluation directory not set.")
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
        
        self.slurm_job.add_command("nequip-evaluate --train-dir ./train_dir --dataset-config eval_config.yaml --metrics-config None --output eval_output.xyz")
        self.slurm_job.write_slurm()
        if dry_run:
            return
        self.slurm_job.submit_slurm()
        self.log_line(f"Submitted job with job id {self.slurm_job.job_id}")

        
        
        