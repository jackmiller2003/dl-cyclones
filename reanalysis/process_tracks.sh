#!/bin/bash
#PBS -q normal
#PBS -P x77
#PBS -N process_tracks
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l jobfs=100GB
#PBS -l walltime=0:30:00
#PBS -l storage=scratch/x77+gdata/x77+gdata/hh5+gdata/rt52
#PBS -l wd

module use /g/data/hh5/public/modules
module load conda/analysis3-22.01

mkdir $PBS_JOBFS/cyclone_binaries

#module load python3/3.9.2

# Run Python application
python3 process_tracks.py $(whoami) > $PBS_JOBID.log
#~/.local/bin/pyinstrument process_tracks.py > $PBS_JOBID-profile.log
