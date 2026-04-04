import os
import numpy as np
from gslibs.utils.coordinates_io import *
import yaml


class FixedColvar_driver:
    def __init__(self, plumed_script_path: str = None): 
        
        
        file_dir = os.path.dirname(os.path.realpath(__file__))
        
        self.plumed_keywords_dir = os.path.join(file_dir, "../../plumed_keywords")
        if not os.path.exists(self.plumed_keywords_dir):
            raise FileNotFoundError(f"Plumed keywords directory {self.plumed_keywords_dir} not found.")
        
        if plumed_script_path is None:
            self.plumed_script = """
# --- INPUTS  ---
d1: DISTANCE ATOMS=8,13
d2: DISTANCE ATOMS=5,9

# --- BIASING  ---
OPES_METAD_EXPLORE ...
LABEL=opes
ARG=d1,d2
FILE=Kernels.data
PACE=10
TEMP=300
BARRIER=50
SIGMA_MIN=0.01,0.01
# RESTART=NO
# STATE_RFILE=compressed_Kernels.data
# STATE_WFILE=compressed_Kernels.data
# STATE_WSTRIDE=2000
# COMPRESSION_THRESHOLD=0.05
... OPES_METAD_EXPLORE

# --- PRINTING  ---
PRINT ARG=d1,d2,opes.* FILE=COLVAR STRIDE=10
        """
        else:
            with open(plumed_script_path, 'r') as f:
                self.plumed_script = f.read()
            
        
    
    def requires_training(self):
        return False
    
    def get_plumed_custom_keywords(self, md_path):
        return []
    
    def get_plumed_section(self):
        return self.plumed_script
    


