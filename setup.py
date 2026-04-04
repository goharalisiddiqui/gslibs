"""
Setup script for gs-utils package.

This setup.py is provided for backward compatibility with older packaging tools.
For modern installations, prefer using pyproject.toml with pip install.
"""

from setuptools import setup, find_packages

# Package requirements
install_requires = [
    "ase>=3.22.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "matplotlib>=3.5.0",
    "pymatgen>=2022.0.0",
]

# Development requirements
extras_require = {
    "dev": [
        "pytest>=6.0",
        "pytest-cov",
        "black",
        "isort",
        "flake8",
        "mypy",
        "pre-commit",
    ],
    "docs": [
        "sphinx>=4.0",
        "sphinx-rtd-theme",
        "numpydoc",
        "myst-parser",
    ],
    "test": [
        "pytest>=6.0",
        "pytest-cov",
        "pytest-mock",
    ],
}

# Complete development requirements
extras_require["all"] = list(set(
    req for extra_reqs in extras_require.values() for req in extra_reqs
))

setup(
    name="gs-utils",
    version="0.1.0",
    description="Computational chemistry and materials science simulation toolkit",
    author="Gohar Ali Siddiqui",
    author_email="goharalisiddiqui@gmail.com",
    url="https://github.com/goharalisiddiqui/gs-utils",
    project_urls={
        "Repository": "https://github.com/goharalisiddiqui/gs-utils.git",
        "Documentation": "https://github.com/goharalisiddiqui/gs-utils",
        "Bug Reports": "https://github.com/goharalisiddiqui/gs-utils/issues",
    },
    packages=find_packages(include=["gslibs", "gslibs.*"]),
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.txt"],
    },
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: System :: Distributed Computing",
    ],
    keywords=[
        "computational-chemistry",
        "materials-science",
        "molecular-dynamics",
        "density-functional-theory",
        "slurm",
        "hpc",
        "lammps",
        "cp2k",
        "orca",
        "plumed",
    ],
    zip_safe=False,
)