#!/bin/bash -l
#PBS -N sentinel_weekly_download_2020
#PBS -A P48500028
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=20GB
#PBS -l walltime=7:00:00
#PBS -q casper
#PBS -j oe

### Set TMPDIR as recommended
export TMPDIR=/glade/derecho/scratch/$USER/temp_serial
mkdir -p $TMPDIR

YEAR='2020'
RESOLUTION='50'

###module swap
ml conda
conda activate proj_2024


###module swap
time python3 sentinel_dataset_weekly_download.py ${YEAR} ${RESOLUTION}

