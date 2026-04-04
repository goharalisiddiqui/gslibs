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
