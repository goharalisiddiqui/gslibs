"""Unit tests for gslibs.drivers.driverutils.slurm."""

import os
import shutil
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from gslibs.drivers.driverutils.slurm import (
    parse_sbatch_header,
    slurm_driver,
    get_job_state,
    SLURM_RUNNING_STATES,
    SLURM_SUCCESS_STATES,
    SLURM_FAILED_STATES,
)


def test_parse_sbatch_header():
    sample_content = """#!/bin/bash
#SBATCH --partition=zencloud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=./slurm_logs/slurm-%J.out
#SBATCH --error=./slurm_logs/slurm-%J.err

module load cuda/12.2
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
source /mnt/projects/venv/bin/activate
"""
    temp_file = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        temp_file.write(sample_content)
        temp_file.flush()
        temp_file.close()

        settings, modules, preamble, env_vars = parse_sbatch_header(temp_file.name)
        assert settings["partition"] == "zencloud"
        assert settings["nodes"] == "1"
        assert settings["cpus-per-task"] == "8"
        assert settings["gres"] == "gpu:1"
        assert settings["mem"] == "16G"
        assert settings["output"] == "./slurm_logs/slurm-%J.out"
        assert "cuda/12.2" in modules
        assert env_vars["CUDA_VISIBLE_DEVICES"] == "0"
        assert any("source /mnt/projects/venv/bin/activate" in line for line in preamble)
    finally:
        os.unlink(temp_file.name)


def test_from_header_file():
    sample_content = """#!/bin/bash
#SBATCH --partition=zencloud
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=./logs/out.log

module load openblas
source /path/to/venv/bin/activate
"""
    temp_dir = tempfile.mkdtemp()
    temp_header = os.path.join(temp_dir, "header.sh")
    try:
        with open(temp_header, "w") as f:
            f.write(sample_content)

        run_dir = os.path.join(temp_dir, "run_1")
        driver = slurm_driver.from_header_file(
            header_path=temp_header,
            job_name="unit_test_job",
            run_dir=run_dir,
            overrides={"cpus-per-task": 8}
        )

        assert driver.job_name == "unit_test_job"
        assert driver.run_dir == run_dir
        assert driver.slurm_config["partition"] == "zencloud"
        assert driver.slurm_config["cpus-per-task"] == 8
        assert "openblas" in driver.modules
        assert any("source /path/to/venv/bin/activate" in line for line in driver.preamble)

        driver.add_command("echo 'Hello World'")
        script_path = driver.write_slurm()
        assert os.path.isfile(script_path)
        assert os.path.isdir(os.path.join(run_dir, "logs"))

        with open(script_path, "r") as f:
            content = f.read()
        assert "#SBATCH --partition=zencloud" in content
        assert "#SBATCH --cpus-per-task=8" in content
        assert "module load openblas" in content
        assert "echo 'Hello World'" in content
    finally:
        shutil.rmtree(temp_dir)


def test_lifecycle_helpers():
    temp_dir = tempfile.mkdtemp()
    try:
        driver = slurm_driver(job_name="test_job", run_dir=temp_dir)
        driver.job_id = "123456"

        with patch("gslibs.drivers.driverutils.slurm.get_job_state") as mock_state:
            mock_state.return_value = "RUNNING"
            assert driver.is_running() is True
            assert driver.is_successful() is False
            assert driver.is_failed() is False

            mock_state.return_value = "COMPLETED"
            assert driver.is_running() is False
            assert driver.is_successful() is True
            assert driver.is_failed() is False

            mock_state.return_value = "FAILED"
            assert driver.is_running() is False
            assert driver.is_successful() is False
            assert driver.is_failed() is True
    finally:
        shutil.rmtree(temp_dir)
