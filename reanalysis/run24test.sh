#!/bin/bash
#PBS -q normal
#PBS -P x77
#PBS -N 24cpusfor5cyclones
#PBS -l ncpus=24
#PBS -l mem=96GB
#PBS -l jobfs=100GB
#PBS -l walltime=0:30:00
#PBS -l storage=scratch/x77+gdata/x77+gdata/hh5+gdata/rt52
#PBS -l wd

#module use /g/data/hh5/public/modules
#module load conda/analysis3-22.01

mkdir $PBS_JOBFS/cyclone_binaries

module load python3/3.9.2

# Run Python application
python3 test_for_cpu.py > $PBS_JOBID-run1.log
python3 test_for_cpu.py > $PBS_JOBID-run2.log
#~/.local/bin/pyinstrument test_for_cpu.py > $PBS_JOBID-profile.log
#python3 test_for_cpu.py > $PBS_JOBID-run2.log
