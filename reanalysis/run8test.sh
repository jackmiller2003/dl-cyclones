#!/bin/bash
#PBS -q normal
#PBS -P x77
#PBS -N 8cpusfor5cyclones
#PBS -l ncpus=8
#PBS -l mem=96GB
#PBS -l jobfs=100GB
#PBS -l walltime=0:30:00
#PBS -l software=python
#PBS -l storage=scratch/x77+gdata/x77+gdata/hh5+gdata/rt52
#PBS -l wd

module use /g/data/hh5/public/modules
module load conda/analysis3-22.01

# Run Python application
python3 /home/156/jm0124/dl-cyclones/reanalysis/test_for_cpu.py > $PBS_JOBID.log
