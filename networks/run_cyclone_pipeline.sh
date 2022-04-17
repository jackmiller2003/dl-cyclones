#!/bin/bash
#PBS -q gpuvolta
#PBS -P x77
#PBS -N cyclone_pipeline
#PBS -l ncpus=24
#PBS -l mem=192GB
#PBS -l jobfs=10GB
#PBS -l walltime=24:00:00
#PBS -l ngpus=2
#PBS -l storage=scratch/x77+gdata/x77+gdata/hh5+gdata/rt52

module use /g/data/hh5/public/modules
module load conda/analysis3-22.01
module load pytorch/1.10.0 

mkdir $PBS_JOBFS/cyclone_binaries

#module load python3/3.9.2

# Run Python application
# this script should be called with:
# $ qsub -v start='0',end='-1' process_all_tracks.sh
# where start, end are indices into the list of tracks
python3 /home/156/jm0124/dl-cyclones/networks/cyclone_pipeline.py > $PBS_JOBID.log
#~/.local/bin/pyinstrument process_tracks.py > $PBS_JOBID-profile.log
