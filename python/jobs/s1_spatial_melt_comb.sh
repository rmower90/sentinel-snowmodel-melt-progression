#!/bin/bash -l
#PBS -N s1_melt_tuolumne
#PBS -A P48500028
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=50GB
#PBS -l walltime=16:00:00
#PBS -q casper
#PBS -j oe

### Set TMPDIR as recommended
export TMPDIR=/glade/derecho/scratch/$USER/temp_serial
mkdir -p $TMPDIR


## module swap
module load conda
conda activate proj_2024

# ## inputs
DOMAIN=tuolumne



time python3 ../scripts/sentinel_spatial_melt.py $DOMAIN

