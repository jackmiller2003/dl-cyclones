#!/bin/bash
#PBS -q normal
#PBS -P x77
#PBS -N process_all_tracks
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l jobfs=10GB
#PBS -l walltime=48:00:00
#PBS -l storage=scratch/x77+gdata/x77+gdata/hh5+gdata/rt52
#PBS -l wd

module use /g/data/hh5/public/modules
module load conda/analysis3-22.01

mkdir $PBS_JOBFS/cyclone_binaries

#module load python3/3.9.2

# Run Python application
# this script should be called with:
# $ qsub -v start='0',end='-1' process_all_tracks.sh
# where start, end are indices into the list of tracks
python3 process_tracks.py $(whoami) ${start} ${end} > $PBS_JOBID.log
#~/.local/bin/pyinstrument process_tracks.py > $PBS_JOBID-profile.log
