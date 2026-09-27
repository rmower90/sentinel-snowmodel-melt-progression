#!/bin/bash -l
#PBS -N sentinel_backscatter_download_2024_similarCues_40m
#PBS -A P48500028
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=20GB
#PBS -l walltime=2:00:00
#PBS -q casper
#PBS -j oe

### Set TMPDIR as recommended
export TMPDIR=/glade/derecho/scratch/$USER/temp_serial
mkdir -p $TMPDIR

#YEAR='2015'
START_DATE='2023-10-01'
END_DATE='2024-10-01'
RESOLUTION='40' # meters

###module swap
ml conda
conda activate proj_2024


###module swap
time python3 sentinel_dataset_backscatter_download.py ${START_DATE} ${END_DATE} ${RESOLUTION}

