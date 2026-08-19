"""SLURM job submission, management, and lifecycle orchestration utilities.

This module provides enterprise-grade functionality for creating, configuring,
submitting, and monitoring SLURM batch jobs across HPC cluster nodes.
"""

import logging
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union

_log = logging.getLogger(__name__)

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
    "gres",
    "account",
    "qos",
    "mail-type",
    "mail-user",
    "constraint",
    "exclusive",
    "array",
]

SLURM_RUNNING_STATES = {
    "PENDING",
    "RUNNING",
    "COMPLETING",
    "CONFIGURING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
}

SLURM_SUCCESS_STATES = {
    "COMPLETED",
}

SLURM_FAILED_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


def parse_sbatch_header(
    header_path: str,
) -> Tuple[Dict[str, str], List[str], List[str], Dict[str, str]]:
    """Parses an SBATCH script or template file for SLURM directives and environment setup.

    Extracts:
    1. SBATCH directives (`#SBATCH --key=value` or `#SBATCH --key value`)
    2. Environment module load commands (`module load <module>`)
    3. Environment variable exports (`export KEY=VALUE`)
    4. Preamble shell commands (e.g., `source /path/to/venv/bin/activate`)

    Parameters
    ----------
    header_path : str
        Path to the script or template file.

    Returns
    -------
    tuple of (dict, list, list, dict)
        - slurm_settings (dict): Dictionary of parsed SBATCH options.
        - modules (list of str): Module names loaded via `module load`.
        - preamble (list of str): Script preamble lines (excluding shebang and sbatch).
        - env_vars (dict): Key-value pairs for environment variables exported.

    Examples
    --------
    >>> settings, modules, preamble, envs = parse_sbatch_header("sbatch_header.sh")
    >>> print(settings.get("partition"))
    'zencloud'
    """
    slurm_settings: Dict[str, str] = {}
    modules: List[str] = []
    preamble: List[str] = []
    env_vars: Dict[str, str] = {}

    if not header_path or not os.path.isfile(header_path):
        return slurm_settings, modules, preamble, env_vars

    with open(header_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#!/bin/bash") or line.startswith("#!/bin/sh") or line.startswith("#! /bin/bash"):
                continue

            if line.startswith("#SBATCH"):
                content = line.replace("#SBATCH", "").strip()
                if content.startswith("--"):
                    content = content[2:]
                if "=" in content:
                    k, v = content.split("=", 1)
                elif " " in content:
                    k, v = content.split(" ", 1)
                else:
                    k, v = content, "true"
                slurm_settings[k.strip()] = v.strip()

            elif line.startswith("module load ") or line.startswith("module add "):
                mod_name = line.split(maxsplit=2)[-1].strip()
                modules.append(mod_name)

            elif line.startswith("export "):
                exp_content = line.replace("export ", "").strip()
                if "=" in exp_content:
                    ek, ev = exp_content.split("=", 1)
                    env_vars[ek.strip()] = ev.strip()
                else:
                    preamble.append(line)

            else:
                preamble.append(line)

    return slurm_settings, modules, preamble, env_vars


def get_job_state(job_id: Union[str, int], max_retries: int = 3) -> str:
    """Queries the current state of a SLURM job using scontrol and sacct.

    Uses regex parsing on `scontrol show job -o <job_id>` for fast inspection,
    with an automatic fallback to `sacct` if the job has completed and exited the queue.

    Parameters
    ----------
    job_id : str or int
        The SLURM job ID to query.
    max_retries : int, optional
        Number of retry attempts if scheduler is briefly unreachable. Default is 3.

    Returns
    -------
    str
        Current job state (e.g., 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'UNKNOWN').

    Examples
    --------
    >>> state = get_job_state("161875")
    >>> print(state)
    'COMPLETED'
    """
    job_id_str = str(job_id).strip()
    if not job_id_str:
        return "UNKNOWN"

    for attempt in range(max_retries):
        try:
            proc = subprocess.run(
                ["scontrol", "show", "job", "-o", job_id_str],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                match = re.search(r"JobState=([A-Za-z_]+)", proc.stdout)
                if match:
                    return match.group(1).upper()
        except Exception as e:
            _log.debug(f"scontrol query attempt {attempt + 1} failed for job {job_id_str}: {e}")

        # Fallback to sacct for completed/historical jobs
        try:
            proc = subprocess.run(
                ["sacct", "-j", job_id_str, "--format=State", "--noheader", "-X"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                tokens = proc.stdout.strip().split()
                if tokens:
                    return tokens[0].upper()
        except Exception as e:
            _log.debug(f"sacct query attempt {attempt + 1} failed for job {job_id_str}: {e}")

        if attempt < max_retries - 1:
            time.sleep(1)

    return "UNKNOWN"


class slurm_driver:
    """SLURM job driver for creating, submitting, and managing computational jobs.

    Provides a clean object-oriented interface to configure SLURM directives,
    manage environment modules and variables, stage execution scripts, submit
    jobs via `sbatch`, monitor lifecycle states, and cancel active jobs.

    Parameters
    ----------
    job_name : str
        Name for the SLURM job.
    run_dir : str
        Absolute path to the directory where the job will execute and write script files.
    slurm_settings : dict, optional
        Dictionary of custom SLURM settings to override defaults. Default is empty dict.

    Attributes
    ----------
    job_name : str
        Name of the SLURM job.
    run_dir : str
        Job execution and script staging directory.
    job_id : str or None
        SLURM job ID assigned after submission.
    slurm_config : dict
        Dictionary of all configured `#SBATCH` directives.
    preamble : list of str
        Shell commands executed prior to module loading and command execution.
    modules : list of str
        Environment modules loaded via `module load`.
    commands : list of str
        Main commands executed by the batch script.
    env_vars : dict
        Environment variables exported prior to command execution.

    Raises
    ------
    ValueError
        If `run_dir` is not an absolute path.
    """

    def __init__(self, job_name: str, run_dir: str, slurm_settings: Optional[Dict[str, Any]] = None):
        self.job_name = str(job_name)
        if not run_dir.startswith("/"):
            raise ValueError(f"SLURM: Run directory must be an absolute path (got: '{run_dir}')")
        self.run_dir = os.path.abspath(run_dir)
        self.slurm_filename = "slurm.sh"
        self.slurm_config: Dict[str, Any] = {}

        # Default standard settings
        self.slurm_config["job-name"] = self.job_name
        self.slurm_config["output"] = "slurm.out"
        self.slurm_config["error"] = "slurm.err"
        self.slurm_config["nodes"] = 1
        self.slurm_config["ntasks"] = 1
        self.slurm_config["cpus-per-task"] = 8
        self.slurm_config["time"] = "10:00:00"
        self.slurm_config["partition"] = "zencloud"
        self.slurm_config["mem"] = "20G"

        if slurm_settings:
            self.update_config(slurm_settings)

        self.job_id: Optional[str] = None
        self.preamble: List[str] = []
        self.modules: List[str] = []
        self.commands: List[str] = []
        self.env_vars: Dict[str, str] = {}

    @classmethod
    def from_header_file(
        cls,
        header_path: str,
        job_name: str,
        run_dir: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "slurm_driver":
        """Factory method creating a fully configured `slurm_driver` from an existing header file.

        Parameters
        ----------
        header_path : str
            Path to an `#SBATCH` header/setup file.
        job_name : str
            Name for the SLURM job.
        run_dir : str
            Absolute path to the job execution directory.
        overrides : dict, optional
            Optional dictionary of SLURM settings to override parsed values.

        Returns
        -------
        slurm_driver
            Configured driver instance with directives, modules, env vars, and preambles set.

        Examples
        --------
        >>> driver = slurm_driver.from_header_file(
        ...     header_path="sbatch_header.sh",
        ...     job_name="trial_0001",
        ...     run_dir="/scratch/trial_0001"
        ... )
        """
        settings, modules, preamble, envs = parse_sbatch_header(header_path)
        if overrides:
            settings.update(overrides)

        driver = cls(job_name=job_name, run_dir=run_dir, slurm_settings=settings)
        for mod in modules:
            driver.add_module(mod)
        for line in preamble:
            driver.add_preamble(line)
        for k, v in envs.items():
            driver.add_env_var(k, v)

        return driver

    def add_preamble(self, line: str) -> None:
        """Appends a command to the batch script preamble (executed before modules and commands)."""
        self.preamble.append(str(line))

    def add_module(self, module: str) -> None:
        """Adds a module to be loaded via `module load <module>`."""
        self.modules.append(str(module))

    def add_command(self, command: str) -> None:
        """Appends an execution command to the batch script."""
        self.commands.append(str(command))

    def add_env_var(self, key: str, value: Any) -> None:
        """Sets an environment variable to be exported in the batch script."""
        self.env_vars[str(key)] = str(value)

    def set_partition(self, partition: str) -> None:
        """Sets the SLURM partition."""
        self.slurm_config["partition"] = str(partition)

    def set_time(self, time_limit: str) -> None:
        """Sets the job execution time limit (e.g., '04:00:00')."""
        self.slurm_config["time"] = str(time_limit)

    def set_mem(self, memory: str) -> None:
        """Sets the job memory allocation (e.g., '16G')."""
        self.slurm_config["mem"] = str(memory)

    def add_config(self, key: str, value: Any) -> None:
        """Adds or overrides a specific `#SBATCH` configuration option."""
        clean_key = str(key).strip().lstrip("-")
        if clean_key not in VALID_SBATCH_KEYS:
            _log.warning(
                f"SLURM: Setting '{clean_key}' is not in standard VALID_SBATCH_KEYS. Adding anyway."
            )
        self.slurm_config[clean_key] = value

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """Updates multiple `#SBATCH` configuration options from a dictionary."""
        for k, v in config_dict.items():
            self.add_config(k, v)

    def _ensure_output_directories(self) -> None:
        """Ensures parent directories for output and error logs exist before submission."""
        for opt in ("output", "error"):
            log_path = self.slurm_config.get(opt)
            if log_path and "/" in log_path:
                parent_dir = os.path.dirname(log_path)
                if not os.path.isabs(parent_dir):
                    parent_dir = os.path.join(self.run_dir, parent_dir)
                os.makedirs(parent_dir, exist_ok=True)

    def write_slurm(self) -> str:
        """Generates and writes the `slurm.sh` batch script file into `run_dir`.

        Returns
        -------
        str
            Absolute path to the written `slurm.sh` script.

        Raises
        ------
        FileNotFoundError
            If `run_dir` does not exist and cannot be created.
        """
        os.makedirs(self.run_dir, exist_ok=True)
        self._ensure_output_directories()

        slurm_filepath = os.path.join(self.run_dir, self.slurm_filename)
        with open(slurm_filepath, "w") as f:
            f.write("#!/bin/bash\n\n")

            for key, value in self.slurm_config.items():
                if value is not None:
                    f.write(f"#SBATCH --{key}={value}\n")
            f.write("\n")

            for line in self.preamble:
                f.write(f"{line}\n")
            if self.preamble:
                f.write("\n")

            for module in self.modules:
                f.write(f"module load {module}\n")
            if self.modules:
                f.write("\n")

            if "OMP_NUM_THREADS" not in self.env_vars and "cpus-per-task" in self.slurm_config:
                self.env_vars["OMP_NUM_THREADS"] = str(self.slurm_config["cpus-per-task"])

            for key, value in self.env_vars.items():
                f.write(f"export {key}={value}\n")
            if self.env_vars:
                f.write("\n")

            for command in self.commands:
                f.write(f"{command}\n")

        return slurm_filepath

    def submit_slurm(self) -> str:
        """Writes the SLURM script and submits it to the cluster queue using `sbatch`.

        Returns
        -------
        str
            Assigned SLURM job ID.

        Raises
        ------
        RuntimeError
            If `sbatch` submission command fails.
        """
        self.write_slurm()

        proc = subprocess.run(
            ["sbatch", self.slurm_filename],
            capture_output=True,
            text=True,
            cwd=self.run_dir,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(
                f"SLURM: Error submitting job {self.job_name} in {self.run_dir}: {proc.stderr}"
            )

        match = re.search(r"Submitted batch job (\d+)", proc.stdout)
        if match:
            self.job_id = match.group(1)
        else:
            self.job_id = proc.stdout.strip().split()[-1]

        return self.job_id

    def check_status(self) -> str:
        """Checks the current lifecycle state of the SLURM job.

        Returns
        -------
        str
            Job state (e.g., 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'UNKNOWN').

        Raises
        ------
        ValueError
            If job has not been submitted yet (job_id is None).
        """
        if self.job_id is None:
            raise ValueError("SLURM: Job ID is not set. Job must be submitted before checking status.")
        return get_job_state(self.job_id)

    def is_running(self) -> bool:
        """Returns True if the job is currently queued or actively executing."""
        return self.check_status() in SLURM_RUNNING_STATES

    def is_successful(self) -> bool:
        """Returns True if the job has completed successfully (state COMPLETED)."""
        return self.check_status() in SLURM_SUCCESS_STATES

    def is_failed(self) -> bool:
        """Returns True if the job has failed, cancelled, or timed out."""
        return self.check_status() in SLURM_FAILED_STATES

    def wait(
        self,
        poll_interval: int = 10,
        timeout: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """Blocks execution and polls the job status until it completes or times out.

        Parameters
        ----------
        poll_interval : int, optional
            Interval in seconds between status polls. Default is 10 seconds.
        timeout : int, optional
            Maximum duration in seconds to wait before raising TimeoutError. Default is None.
        verbose : bool, optional
            If True, logs periodic status updates. Default is False.

        Returns
        -------
        str
            Final job state (e.g., 'COMPLETED', 'FAILED', 'TIMEOUT').

        Raises
        ------
        TimeoutError
            If execution exceeds `timeout` seconds.
        """
        start_time = time.time()
        poll_interval = max(int(poll_interval), 2)
        time.sleep(min(poll_interval, 3))

        while True:
            state = self.check_status()
            if verbose:
                _log.info(f"Job {self.job_id} ({self.job_name}) state: {state}")

            if state not in SLURM_RUNNING_STATES and state != "UNKNOWN":
                return state

            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"SLURM: Job {self.job_id} ({self.job_name}) exceeded timeout of {timeout}s"
                )

            time.sleep(poll_interval)

    def cancel_job(self) -> str:
        """Cancels the active SLURM job using `scancel`.

        Returns
        -------
        str
            Output from `scancel` command.

        Raises
        ------
        ValueError
            If job ID is not set.
        RuntimeError
            If `scancel` command fails.
        """
        if self.job_id is None:
            raise ValueError("SLURM: Job ID is not set. Cannot cancel unsubmitted job.")
        proc = subprocess.run(
            ["scancel", str(self.job_id)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"SLURM: Error cancelling job {self.job_id}: {proc.stderr}")
        return proc.stdout.strip()
