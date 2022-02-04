import json
import h5py

import numpy as np
import os
from PIL import Image
from scipy.ndimage import zoom
import torch
from torch.utils.data import DataLoader, random_split
from scipy.spatial.transform import Rotation as R
CUDA = torch.cuda.is_available()

KWARGS = {"num_workers": 1, "pin_memory": True} if CUDA else {}


def initialization_path(cuda):
    if cuda:
        cryo_train_val_dir = os.getcwd() + "/Cryo/vae_pipeline/"
    else:
        cryo_train_val_dir = os.getcwd() + "\\"
    return cryo_train_val_dir


def hinted_tuple_hook(obj):
    """Transform a list into tuple.

    Parameters
    ----------
    obj : *
        Value of a dic.

    Returns
    -------
    tuple,
        Transform the value of a dic into dic.
    obj : *
        Value of a dic.
    """
    if "__tuple__" in obj:
        return tuple(obj["items"])
    return obj
    if "__none__" in obj:
        return None


def load_meta_config(path):
    with open(path) as json_file:
        meta_config = json.load(json_file, object_hook=hinted_tuple_hook)
        meta_config["dataset_name"] = meta_config["dataset"] + \
            "_" + str(meta_config["size"])
        return meta_config


def join_dics(list_dics):
    new_dic = {}
    for dic in list_dics:
        keys = dic.keys()
        for key in keys:
            new_dic[key] = dic[key]
    return new_dic


def load_nd_config(path):
    with open(path) as json_file:
        nd_config = json.load(json_file, object_hook=hinted_tuple_hook)
        vae = nd_config["vae"]
        losses = nd_config["losses"]
        cnn = nd_config["cnn"]
        wigner_representation = nd_config["wigner_representation"]
        latent_space = nd_config["latent_space"]
        datasets = nd_config["datasets"]
        config = join_dics(
            [vae, losses, cnn, wigner_representation, latent_space, datasets])
        return config


def choose_dimension(meta_config, path):
    ndim = meta_config["dimension"]
    if ndim == 1:
        config = load_nd_config(path+"1d_config.json")
        config = join_dics([config, meta_config])
        config["img_shape"] = (1,)+(meta_config["size"],
                                    )*meta_config["dimension"]
        return config
    if ndim == 2:
        config = load_nd_config(path+"2d_config.json")
        config = join_dics([config, meta_config])
        config["img_shape"] = (1,)+(meta_config["size"],
                                    )*meta_config["dimension"]
        return config
    config = join_dics([config, meta_config])
    config = load_nd_config(path+"3d_config.json")
    config["img_shape"] = (1,)+(meta_config["size"],)*meta_config["dimension"]
    return config


def normalize_torch(dataset, scale="linear"):
    """Normalize a tensor.

    Parameters
    ----------
    dataset : torch tensor
        Images.
    scale : string
        Methods of normalization.

    Returns
    -------
    dataset : torch tensor
        Normalized images.
    """
    if scale == "linear":
        for i, data in enumerate(dataset):
            min_data = torch.min(data)
            max_data = torch.max(data)
            if max_data == min_data:
                raise ZeroDivisionError
            dataset[i] = (data - min_data) / (max_data - min_data)
    return dataset


def initialization_dataset(meta_config, path):
    if not os.path.exists(path):
        raise OSError
    if path.lower().endswith(".h5"):
        data_dict = h5py.File(path, "r")
        all_datasets = data_dict["particles"][:]
    else:
        all_datasets = np.load(path)
    dataset = np.asarray(all_datasets)
    img_shape = dataset.shape
    img_size = img_shape[-1]
    n_imgs = img_shape[0]
    new_dataset = []
    dimension = meta_config["dimension"]
    dataset = torch.Tensor(dataset)
    size = meta_config["size"]
    if dimension == 1:
        if len(img_shape) == 2:
            dataset = dataset.reshape((n_imgs, 1, img_size))
        zoom_coef = [size/img_size]
        for i in range(n_imgs):
            image = zoom(dataset[i][0], zoom_coef)
            new_dataset.append(np.asarray(image))
    if dimension == 2:
        if len(img_shape) == 3:
            dataset = dataset.reshape((n_imgs, 1, img_size, img_size))
            dataset = np.asarray(dataset)
        for i in range(n_imgs):
            image = Image.fromarray(dataset[i][0]).resize([size, size])
            new_dataset.append(np.asarray(image))
    elif dimension == 3:
        if len(img_shape) == 4:
            dataset.reshape((n_imgs, 1, img_size, img_size, img_size))
        zoom_coef = [size/img_size for i in range(3)]
        for i in range(n_imgs):
            image = Image.fromarray(dataset[i][0]).resize([size, size, size])
            new_dataset.append(np.asarray(image))

    dataset = torch.Tensor(new_dataset)
    dataset = normalize_torch(dataset)
    dataset = dataset.reshape((n_imgs, 1) + (size,)*dimension)

    return dataset


def split_dataset(dataset, config):
    """Separate data in train and validation sets.

    Parameters
    ----------
    dataset : torch tensor
        Images.
    batch_size : int
        Batch_size.
    frac_val : float
        Ratio between validation and training datasets.

    Returns
    -------
    trainset : tensor
        Training images.
    testset : tensor
        Test images.
    trainloader : tensor
        Ready to be used by the NN for training images.
    testloader : tensor
        Ready to be used by the NN for test images.
    """
    batch_size = config["batch_size"]
    frac_val = config["frac_val"]
    n_imgs = len(dataset)
    n_val = int(n_imgs * frac_val)
    trainset, testset = random_split(dataset, [n_imgs - n_val, n_val])

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **KWARGS)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, **KWARGS)
    return trainset, testset, trainloader, testloader


def open_labels(path):
    """
    Open the file containing grown truth

    Parameters
    ----------
    path : string
        path of the dataset labels.

    Returns
    -------
    rotation : array
        Representation of rotation into matrices.
    rotvec : array
        Representation of rotation into axis angle.
    labels : array
        Representation of rotation into quaternion.
    euler : array
        Representation of rotation into euler ZYZ.

    """
    labels_lec = np.load(path, allow_pickle=True)
    if len(labels_lec.shape) < 4:
        return labels_lec, labels_lec, labels_lec, labels_lec
    labels1 = labels_lec.T[0].T
    n = labels1.shape[0]
    labels = np.zeros((n, 4))
    for i in range(n):
        labels[i] = labels1[i]
    rot_representation = R.from_quat(labels)
    rotation = rot_representation.as_matrix()
    rotvec = rot_representation.as_rotvec()
    euler = rot_representation.as_euler("ZYZ")
    return rotation, rotvec, labels, euler
