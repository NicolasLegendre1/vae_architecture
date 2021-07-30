#!/bin/sh
 
#
#PBS -l walltime=48:00:00,select=1:ncpus=16:mem=40gb:ngpus=4
#PBS -N Run_3D_Cryo
#PBS -A pr-kdd-1-gpu
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
export GEOMSTATS_BACKEND="pytorch"
python3 Cryo/VAE_Cryo_V3/train.py 
#python3 Cryo/VAE_Cryo_V3/ResultsAnalyses.py

