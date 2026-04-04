import os
import numpy as np
import yaml

from gsutils.utils.io_coordinates import *

from gsutils.drivers.base import base_driver

class MLcolvar_driver(base_driver):
    def __init__(self, input_file, 
                 run_name: str, 
                 run_dir: str,
                 verbose=False,
                 meta: dict = {}): 
        super().__init__(input_file, run_name, run_dir, verbose, meta)
        
        
        self.slurm_settings = {"partition": "carlos", 
                               "time": "3:00:00", 
                               "mem": "24G",
                               "ntasks": 1,
                               "cpus-per-task": 8}
                            #    "gres": "gpu:1",
        
        file_dir = os.path.dirname(os.path.realpath(__file__))
        venv_activate = os.path.join(file_dir, "../../.venv/bin/activate")
        venv_activate = os.path.abspath(venv_activate)
        self.slurm_commands.append("source {}".format(venv_activate))
        
        self.plumed_keywords_dir = os.path.join(file_dir, "../../plumed_keywords")
        if not os.path.exists(self.plumed_keywords_dir):
            raise FileNotFoundError(f"Plumed keywords directory {self.plumed_keywords_dir} not found.")
            
        
        self.ce_engine_path = os.path.abspath(os.path.join(file_dir, "collective_encoder/engine.py"))
        self.train_data = None
    
    def requires_training(self):
        return True
    
    def additional_load_from_module(self, module):
        self.train_data = module.train_data
    
    def set_train_data(self, train_data):
        assert train_data is not None, "Training data cannot be None."
        assert os.path.exists(train_data), f"Training data {train_data} not accessible."
        assert os.path.isfile(train_data), f"Training data {train_data} is not a file."
        assert train_data.startswith("/"), f"Training data {train_data} is not an absolute path."
        assert train_data.endswith(".xyz"), f"Training data {train_data} is not a .xyz file."
        self.train_data = train_data
    
    def run(self, dry_run=False):
        if self.train_data is None:
            raise ValueError("No training data provided.")
        super().run(dry_run=dry_run)
    
    def prerun(self):
        comm = f"python {self.ce_engine_path} "
        input_conf = "\n".join(self.input_script)
        conf = yaml.safe_load(input_conf)
        for key,val in conf.items():
            if key == "flags":
                for flag in val:
                    comm += f"--{flag} "
            else:
                comm += f"--{key} '{val}' "
        comm += f"--outpath {self.run_dir} "
        comm += f"--save_serial_model_path .. "
        comm += f"--xyzfile {self.train_data} "
        comm += f"--save_serial_model "
        self.slurm_job.add_command(comm)

    def get_plumed_custom_keywords(self, md_path):
        plumed_keyword_dir = os.path.abspath(self.plumed_keywords_dir)
        plumed_cutom_keywords = []
        for keyword_file in ['CARTNN/CartNN.cpp']:
            plumed_cutom_keywords.append(os.path.join(plumed_keyword_dir, keyword_file))
        return plumed_cutom_keywords
    
    def get_plumed_section(self):
        plumed_script = """
# --- INPUTS  ---
WHOLEMOLECULES STRIDE=1 ENTITY0=system
s: CARTNN MODEL=encoder.pt ATOMS=system

# --- BIASING  ---
OPES_METAD_EXPLORE ...
LABEL=opes
ARG=s.node-0
FILE=Kernels.data
PACE=2000
TEMP=300
BARRIER=50
SIGMA_MIN=0.01
# RESTART=NO
# STATE_RFILE=compressed_Kernels.data
# STATE_WFILE=compressed_Kernels.data
# STATE_WSTRIDE=2000
# COMPRESSION_THRESHOLD=0.05
... OPES_METAD_EXPLORE

# --- PRINTING  ---
PRINT ARG=s.*,opes.* FILE=COLVAR STRIDE=1
        """
        return plumed_script
    
    def get_encoder_path(self):
        return os.path.join(self.run_dir, "encoder.pt")


