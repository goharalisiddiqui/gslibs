"""
Filesystem utilities for safe file operations.

This module provides utility functions for safe file operations including
backup and move operations.
"""

import shutil
import os

def backup_move(entity):
    """
    Safely move a file or directory by creating a backup with incremental naming.

    Creates backup copies with names like 'entity.bak1', 'entity.bak2', etc.
    to avoid overwriting existing backups.

    Parameters
    ----------
    entity : str
        Path to the file or directory to backup and move.
    """
    i = 1
    while os.path.exists(entity + f".bak{i}"):
        i += 1
    backup = entity + f".bak{i}"
    shutil.move(entity, backup)

def create_rundir(path: str, stem: str, nexp: str, overwrite=False) -> str:
    '''
    Create a run directory with incremental naming to avoid overwriting existing runs.


    Parameters    
    ----------
    path : str
        Base path where the run directory will be created.
    stem : str
        Stem for the run directory name.
    nexp : str
        Experiment number to be appended to the stem for the run directory name.
    overwrite : bool, optional
        If True, overwrite the existing run directory if it exists. Default is False.
        
    Returns
    -------
    str
        The path to the created run directory.
    '''

    workdir = os.path.join(path, "_".join([stem, str(nexp)]))
    if not overwrite:
        while True:
            if not os.path.isdir(workdir):
                os.makedirs(workdir)
                break
            nexp = nexp + 1
            workdir = os.path.join(path, "_".join([stem, str(nexp)]))
    else:
        if not os.path.isdir(workdir):
            os.makedirs(workdir)

    if len(os.listdir(workdir)) != 0:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
        os.mkdir(workdir)
    return workdir

def output_to_file(workdir, filename="out.txt"):
    '''
    Redirect standard output to a file in the specified directory.

    Parameters
    ----------
    workdir : str
        Directory where the output file will be created.
    filename : str, optional
        Name of the output file. Default is "out.txt".
    '''
    import sys
    import subprocess
    print(f"Redirecting output to file {os.path.join(workdir, filename)}")
    tee = subprocess.Popen(
        ["tee", os.path.join(workdir, filename)], stdin=subprocess.PIPE)
    # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
    # of any child processes we spawn)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())