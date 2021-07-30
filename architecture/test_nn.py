"""Test nn.py"""

import nn


def test():
    CUDA = torch.cuda.is_available()
    if CUDA:
        path_vae = "Cryo/VAE_Cryo_V3/vae_parameters.json"
        path_data = "Cryo/VAE_Cryo_V3/data_parameters.json"
    else:
        path_vae = "vae_parameters.json"
        path_data = "data_parameters.json"
    PATHS, SHAPES, CONSTANTS, SEARCH_SPACE, _ = ds.load_parameters(
        path_vae)
    CONSTANTS.update(SEARCH_SPACE)
    CONSTANTS["latent_space_definition"] = 1
    CONSTANTS["latent_dim"] = 6
    enc = EncoderConv(CONSTANTS)
    A = torch.zeros(200, 1, 64, 64, 64)
    B = enc.forward(A)
    print(B[2].shape)
    dec = DecoderConv(CONSTANTS)
    C = dec.forward(B[2])
    return C
