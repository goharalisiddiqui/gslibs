# GS-Utils: Computational Chemistry and Materials Science Toolkit

A Python package providing drivers and utilities for running computational chemistry and materials science simulations through SLURM job scheduler.

## Features

- **Multi-engine support**: Drivers for LAMMPS, CP2K, ORCA, and more
- **SLURM integration**: Seamless job submission and monitoring
- **Enhanced sampling**: PLUMED integration for advanced MD techniques
- **File format support**: Comprehensive coordinate file I/O (XYZ, PDB, LAMMPS, etc.)
- **Modular design**: Clean separation of drivers and utilities

## Installation

### Prerequisites

- Python 3.7+
- ASE (Atomic Simulation Environment)
- NumPy
- Pandas
- Matplotlib
- PyMatGen (for specific features)

### Install from source

```bash
git clone <repository_url>
cd gs-utils
pip install -e .
```

### Dependencies

The package requires several Python dependencies that are automatically installed:

```
ase >= 3.22.0
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.5.0
pymatgen >= 2022.0.0
```

For HPC environments with module systems, load the appropriate modules:

```bash
module load python/3.12
```

## Quick Start

### Basic LAMMPS Simulation

```python
from gslibs.drivers.mdengines.lammps import lammps_driver

# Initialize driver
lmp = lammps_driver(
    input_file="/path/to/input.in",
    run_name="my_simulation",
    run_dir="/path/to/run_directory"
)

# Add coordinate files
lmp.add_coordinates_file("initial.xyz", "start.lammps")

# Submit job
lmp.run()

# Monitor status
while lmp.is_job_running():
    time.sleep(30)

# Extract results
if lmp.check_status() == "COMPLETED":
    lmp.create_fail_frame("COLVAR", threshold=0.5)
```

### CP2K Single-Point Calculation

```python
from gslibs.drivers.dftengines.cp2k import cp2k_driver

# Initialize driver
cp2k = cp2k_driver(
    input_file="/path/to/cp2k.inp",
    run_name="dft_calc",
    run_dir="/path/to/run_directory"
)

# Add structure file
cp2k.add_coordinates_file("structure.xyz", "input.xyz")

# Submit calculation
cp2k.run()

# Extract final structure with forces
cp2k.write_final_singlepoint("final_structure.xyz")
```

### Coordinate File Conversion

```python
from gslibs.utils.coordinates_io import *

# Read various formats
atoms = any_to_ase("input.pdb")
atoms = xyz_to_ase("structure.xyz")
atoms = lammpsdata_to_ase("data.lammps")

# Convert between formats
xyz_content = ase_to_xyz(atoms)
pdb_content = ase_to_pdb(atoms)
lammps_content = ase_to_lammpsdata(atoms, {"specorder": ["C", "H", "O"]})
```

### PLUMED Output Analysis

```python
from gslibs.utils.plumed import plumed_outfile_reader, plumed_outfile_plotter

# Read PLUMED output
data = plumed_outfile_reader("COLVAR", columns=["CV", "metad.bias"])

# Plot time series
plumed_outfile_plotter("COLVAR", columns=["CV"], title="Collective Variable")
```

## Package Structure

```
gs-utils/
├── gslibs/
│   ├── drivers/           # Simulation drivers
│   │   ├── base.py        # Base driver class
│   │   ├── dftengines/    # DFT calculation drivers
│   │   │   ├── cp2k.py    # CP2K driver
│   │   │   └── orca.py    # ORCA driver
│   │   ├── mdengines/     # MD simulation drivers
│   │   │   ├── lammps.py  # LAMMPS driver
│   │   │   └── gromacs.py # Gromacs driver
│   │   ├── mlmodels/      # ML model drivers
│   │   ├── colvarengines/ # Collective variable engines
│   │   └── driverutils/   # Utility drivers
│   │       └── slurm.py   # SLURM interface
│   └── utils/             # Utility functions
│       ├── coordinates_io.py  # File format conversions
│       ├── plumed.py      # PLUMED output parsing
│       ├── common.py      # Common utilities
│       ├── filesystem.py  # File operations
│       └── orca.py        # ORCA output parsing
├── README.md
├── pyproject.toml
└── setup.py
```

## API Reference

### Drivers

#### Base Driver (`base_driver`)

All simulation drivers inherit from `base_driver`, providing common functionality:

- **File management**: Add coordinate, auxiliary, and binary files
- **Job control**: Submit, monitor, and cancel SLURM jobs
- **Template processing**: Dynamic input file modification
- **Status monitoring**: Real-time job status checking

Key methods:
- `add_coordinates_file(source, dest)`: Add coordinate files
- `add_aux_file(source, dest)`: Add auxiliary files
- `template_input_script(template)`: Apply template substitutions
- `run(dry_run=False)`: Submit job
- `check_status()`: Get job status
- `is_job_running()`: Check if job is active

#### LAMMPS Driver (`lammps_driver`)

Specialized driver for LAMMPS molecular dynamics simulations:

- **Enhanced sampling**: PLUMED integration
- **Trajectory analysis**: Automatic fail frame detection
- **Format support**: LAMMPS data/dump files

Unique methods:
- `set_enhanced_sampler(sampler)`: Configure enhanced sampling
- `create_fail_frame(method, **kwargs)`: Extract failed configurations
- `get_completed_nsteps()`: Get simulation progress

#### DFT Drivers (`cp2k_driver`, `orca_driver`)

Drivers for quantum chemistry calculations:

- **Energy extraction**: Parse final energies and forces
- **Unit conversion**: Automatic Hartree ↔ eV conversion
- **Structure output**: Generate final structures with properties

Unique methods:
- `write_final_singlepoint(dest, source_structure)`: Extract final results

### Utilities

#### Coordinate I/O (`coordinates_io`)

Universal coordinate file format support:

- `any_to_ase(file)`: Auto-detect and read any supported format
- `ase_to_xyz(atoms, extxyz=True)`: Write XYZ files
- `ase_to_pdb(atoms)`: Write PDB files
- `ase_to_lammpsdata(atoms, extra_info)`: Write LAMMPS data files

#### PLUMED Utilities (`plumed`)

PLUMED output file analysis:

- `plumed_outfile_reader(file, columns)`: Parse PLUMED output
- `plumed_outfile_plotter(file, columns)`: Visualize time series data

#### Common Utilities (`common`)

General-purpose functions:

- `wait_for_runs(runs)`: Wait for multiple jobs to complete
- `parse_vars(items)`: Parse command-line key=value pairs
- `get_colors(n, colormap)`: Generate color lists from matplotlib colormaps

## Configuration

### SLURM Settings

Default SLURM settings can be customized per driver:

```python
slurm_settings = {
    "partition": "gpu",
    "time": "24:00:00",
    "mem": "64G",
    "ntasks": 8,
    "cpus-per-task": 4,
    "gres": "gpu:1"
}

driver = lammps_driver(
    input_file="input.in",
    run_name="simulation",
    run_dir="/path/to/run",
    slurm_settings=slurm_settings
)
```

### Environment Modules

For HPC systems using environment modules:

```python
driver.slurm_modules = ['python/3.12', 'lammps/2023.08.02']
driver.runtime_env = {"OMP_NUM_THREADS": "8"}
```

## Examples

### Enhanced Sampling with PLUMED

```python
from gslibs.drivers.mdengines.lammps import lammps_driver

# Setup LAMMPS with PLUMED
lmp = lammps_driver(input_file="md.in", run_name="metad", run_dir="./run")
lmp.add_coordinates_file("start.xyz", "initial.lammps")
lmp.add_aux_file("plumed.dat", "plumed.dat")

# Template PLUMED input
lmp.template_aux_file("plumed.dat", {
    "TEMP": "300",
    "BIASFACTOR": "10"
})

# Run simulation
lmp.run()

# Analyze results
if lmp.check_status() == "COMPLETED":
    # Extract fail frame when CV > 0.8
    lmp.create_fail_frame("COLVAR", 
                         metric="CV", 
                         criterion_type="greater", 
                         threshold=0.8)
```

### Multi-Engine Workflow

```python
from gslibs.drivers.dftengines.cp2k import cp2k_driver
from gslibs.drivers.mdengines.lammps import lammps_driver
from gslibs.utils.common import wait_for_runs

# Initial structure optimization with CP2K
opt = cp2k_driver("optimize.inp", "opt", "./cp2k_opt")
opt.add_coordinates_file("initial.xyz", "structure.xyz")
opt.run()

# Wait for optimization
wait_for_runs([opt])

# Extract optimized structure
opt.write_final_singlepoint("optimized.xyz")

# Run MD simulation with optimized structure
md = lammps_driver("md.in", "production", "./md_run")
md.add_coordinates_file("optimized.xyz", "start.lammps")
md.run()
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use gs-utils in your research, please cite:

```bibtex
@software{gsutils2024,
  title = {GS-Utils: Computational Chemistry and Materials Science Toolkit},
  author = {Seemab, Gohar},
  year = {2024},
  url = {https://github.com/username/gs-utils}
}
```

## Support

For questions, issues, or feature requests:

- **GitHub Issues**: [Create an issue](https://github.com/username/gs-utils/issues)
- **Email**: gohar.seemab@oist.jp

## Changelog

### Version 0.1.0
- Initial release
- LAMMPS, CP2K, and ORCA drivers
- SLURM job management
- Coordinate file I/O utilities
- PLUMED integration
- Enhanced sampling support