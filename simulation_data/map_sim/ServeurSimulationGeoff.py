# -*- coding: utf-8 -*-
"""
Created on Thu May  6 23:16:30 2021

@author: NLEGENDN
"""

import os

import dataset as ds
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
import simulate as sm
import torch


def doplt(arr2d, cmap="gray", **kwargs):
    plt.imshow(arr2d, cmap=cmap, **kwargs)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

paths = {}
paths[
    "3dsimulated"
] = "C:\\Users\\NLEGENDN\\Desktop\\Travaux_Vancouver\\Code_Python\\CryoEM\VAE_Cryo_V3\\Data\\EMD-8441\\map\\emd_8441.map"
paths[
    "3dsimulated"
] = "C:\\Users\\NLEGENDN\\Desktop\\Travaux_Vancouver\\Code_Python\\CryoEM\VAE_Cryo_V3\\Data\\emd_22085.map"
paths[
    "3dsimulated"
] = "/arc/project/ex-kdd-1/NicolasLegendre/Cryo/VAE_Cryo_V3/Data/emd_22085.map"

fname = paths["3dsimulated"]
map_mrc = mrcfile.open(fname)
map_original = map_mrc.data
new_size = 55
size = map_original.shape[0]
# map_original = map_original[new_size:size-new_size,
#                            new_size:size-new_size, new_size:size-new_size]
N = map_original.shape[0]
psize_original = map_mrc.voxel_size.item(0)[0]
doplt(map_original.sum(0))
print("OK")


A = sm.simulate_slice(
    map_r=map_original,
    psize=0.5,
    n_particles=10,
    N_crop=None,
    snr=0.1,
    do_snr=False,
    do_ctf=True,
    df_min=15000,
    df_max=15000,
    df_diff_min=0,
    df_diff_max=0,
    df_ang_min=0,
    df_ang_max=360,
    kv=300,
    cs=2.0,
    ac=0.1,
    phase=0,
    bf=0,
    do_log=True,
    random_seed=0,
)

NoNoise = torch.Tensor(A[0])
NoNoise = ds.normalization_linear(NoNoise)
img_shape = NoNoise.shape
NoNoise = NoNoise.reshape((img_shape[0], 1, img_shape[1], img_shape[1]))

Noise = torch.Tensor(A[1])
Noise = ds.normalization_linear(Noise)
img_shape = Noise.shape
Noise = Noise.reshape((img_shape[0], 1, img_shape[1], img_shape[1]))


np.save(
    "/scratch/ex-kdd-1/NicolasLegendre/Cryo/Data/simulationGeoffNoNoise.npy", NoNoise
)
np.save("/scratch/ex-kdd-1/NicolasLegendre/Cryo/Data/simulationGeoffNoise.npy", Noise)
A[2].to_csv("/scratch/ex-kdd-1/NicolasLegendre/Cryo/Data/simulationGeoffMeta.csv")
