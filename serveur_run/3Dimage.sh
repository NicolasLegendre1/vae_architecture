#!/bin/sh
 
#
#PBS -l walltime=48:00:00,select=1:ncpus=16:mem=40gb
#PBS -N 3D_images
#PBS -A pr-kdd-1
#PBS -m abe
#PBS -M nicolas.legendre@student-cs.fr
#PBS -o /scratch/pr-kdd-1/NicolasLegendre/output.txt
#PBS -e /scratch/pr-kdd-1/NicolasLegendre/error.txt
 
################################################################################
 
# load your modules
module load gcc
module load cuda
module load git
module load python3




cd /arc/project/pr-kdd-1/NicolasLegendre/
#consume your envs, activate environments
source Cryo/bin/activate
python3 Cryo/VAE_Cryo_V3/sim_volumes.py

