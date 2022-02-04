"""Simulate datasets."""
import numpy as np
from PIL import Image
import mrcfile


def open_image(path, size):
    image = Image.open(path)
    image = image.resize((size, size))
    image = np.asarray(image)[:, :, 0]
    return image


def open_volume(path, size):
    volume = mrcfile.open(path)
    volume = volume.data
    volume = volume.resize((size, size, size))
    return volume


def simulate_image(size):
    image = np.zeros((size, size))
    center_cir = np.asarray([int(size/2 + size / 4), int(size/2 + size / 4)])
    center_rec = np.asarray([int(size/2 - size / 4), int(size/3 + size / 4)])
    center_tri = np.asarray([int(size/3 + size / 4), int(size/2 - size / 4)])
    rayon_circ = size/8
    h = size/8
    l = size/4
    for i in range(size):
        for j in range(size):
            if is_in_figure(np.asarray([i, j]), {"circ": [(center_cir, rayon_circ)], "rec": [(center_rec, rayon_circ)], "tri": [(center_tri, [h, l/2])]}):
                image[i][j] = 1
    return image


def distance_L2(point1, point2):
    """
    Calculate the L2 distance of 2 vectors of R^2.

    Parameters
    ----------
    point1 : array of size [2]
    point2 : array of size [2]

    Returns
    -------
    float : distance between point1 and point2.


    """
    return ((point1-point2)**2).sum()


def distance_rec(point1, point2):
    """
    Calculate the L1 distance of 2 vectors of R^2.

    Parameters
    ----------
    point1 : array of size [2]
    point2 : array of size [2]

    Returns
    -------
    float : distance between point1 and point2.

    """
    return abs((point1-point2)).max()


def is_in_circle(point1, point2, rayon):
    return distance_L2(point1, point2) <= rayon**2


def is_in_rec(point1, point2, rayon):
    return distance_rec(point1, point2) <= rayon


def is_in_tri(point1, point2, dims):
    if not is_in_rec(point1, point2, dims[0]):
        return False
    dif = (point1-point2)
    if dif[1] > 0:
        return False
    a = abs(dif[0])/dims[0]
    b = abs(dif[1])/dims[1]
    return a <= b


def is_in_figure(point, centers):
    for center_cir, rayon in centers["circ"]:
        if is_in_circle(point, center_cir, rayon):
            return True
    for center_rec, rayon in centers["rec"]:
        if is_in_rec(point, center_rec, rayon):
            return True
    for center_tri, rayon in centers["tri"]:
        if is_in_tri(point, center_tri, rayon):
            return True
    else:
        return False


def save_image(image, path):
    np.save(path, image)
