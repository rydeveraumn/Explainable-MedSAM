from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = [line.strip() for line in f.read().splitlines() if line.strip() and not line.startswith('#')]

setup(
    name="xai_medsam",
    version="0.1",
    author="Ryan Devera, Corey Senger",
    description="An explainability framework for analyzing MedSAM",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rydeveraumn/Explainable-MedSam",
    packages=find_packages(),
    install_requires=required
)