#!/bin/bash -l
# PBS -N S1-bscatter-melt-thresh-tuolumne-2017
# PBS -A P48500028
# PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=50GB
# PBS -l walltime=14:00:00
# PBS -q casper
# PBS -j oe

### Set TMPDIR as recommended
export TMPDIR=/glade/derecho/scratch/$USER/temp_serial
mkdir -p $TMPDIR

#YEAR='2015'
START_DATE='2016-10-01'
END_DATE='2017-10-01'
RESOLUTION='100' # meters
DOMAIN='tuolumne'

###module swap
ml conda
conda activate proj_2024


###module swap
time python3 sentinel_dataset_weekly_download.py ${START_DATE} ${END_DATE} ${RESOLUTION} ${DOMAIN}

