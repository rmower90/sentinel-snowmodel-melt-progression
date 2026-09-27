#!/bin/bash -l
#PBS -N sm_melt_tuolumne_2017_corrected_wt_0.00035
#PBS -A P48500028
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=100GB
#PBS -l walltime=06:00:00
#PBS -q casper
#PBS -j oe

### Set TMPDIR as recommended
export TMPDIR=/glade/derecho/scratch/$USER/temp_serial
mkdir -p $TMPDIR


## module swap
module load conda
conda activate proj_2024

model='corrected'
wy=2017
method='revamp'
water_surface=0.00035

echo "Running SnowModel spatial melt classification with the following parameters:"
echo "Model: $model"
echo "Water Year: $wy"
echo "Method: $method"
echo "Water Threshold (Surface): $water_surface"



time python3 ../scripts/sm_spatial_melt.py --model $model --wy $wy --method $method --water-thresh-surface $water_surface --workers 1

