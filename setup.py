"""
Pip-installable package for src module.

To install, in the root directory, run: `pip install -e .`
"""

from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
)
