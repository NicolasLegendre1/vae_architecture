import numpy as np
from skimage.transform._warps import rotate


def open_image(path):
    return np.load(path)


def generate_1d_rotations(N_rots):
    return 360*np.random.rand(N_rots)


def simulate_image_rotations(N_rots, path_image):
    image = open_image(path_image)
    rotations = generate_1d_rotations(N_rots)
    size = image.shape[-1]
    images = np.zeros((N_rots, size, size))
    for i in range(N_rots):
        images[i] = rotate(image, rotations[i])
    return (images, rotations)


def projection_image_lines(images):
    N_rots, size, size = images.shape
    lines = np.zeros((N_rots, size))
    for i in range(N_rots):
        lines[i] = images[i].sum(axis=0)
    return lines


def simulate_datasets(N_rots, path_image):
    images, rotations = simulate_image_rotations(N_rots, path_image)
    lines = projection_image_lines(images)
    return images, rotations, lines


def save_simulated_dataset(images, rotations, lines, path_dataset):
    np.save(path_dataset+"images.npy", images)
    np.save(path_dataset+"lines.npy", lines)
    np.save(path_dataset+"rotations.npy", rotations)


def simulate_dataset(N_rots, path_image, path_dataset):
    images, rotations, lines = simulate_datasets(N_rots, path_image)
    save_simulated_dataset(images, rotations, lines, path_dataset)
    return images, rotations, lines
