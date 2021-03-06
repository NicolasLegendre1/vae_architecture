"""Visualization tools for pca."""

import imageio
import functools
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import train_utils
import initialization_pipeline as inipip
import from_vae_to_latent_space as fvlp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

A = np.random.rand(64, 64, 64)
CMAPS_DICT = {'vae': 'Reds', 'iwae': 'Oranges', 'vem': 'Blues'}
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

num = 31
colormap = cm.get_cmap('viridis')
COLORS_FOCUS = colormap(np.linspace(start=0, stop=1, num=num))
num = 2 * 180 + 1
colormap = cm.get_cmap('twilight')
COLORS_THETA = colormap(np.linspace(start=0, stop=1, num=num))
COLORS_Y = colormap(np.linspace(start=0, stop=1, num=181))
COLORS_ROTVEC = colormap(np.linspace(start=0, stop=1, num=181))
COLORS_SUM = colormap(np.linspace(start=0, stop=1, num=360))
COLORS_QUAT = colormap(np.linspace(start=0, stop=1, num=101))

COLORS_X_MOD = colormap(np.linspace(start=0, stop=1, num=45))

COLORS = {
    'rotation_z1': COLORS_THETA,
    'rotation_z2': COLORS_THETA,
    'rotation_y': COLORS_Y,
    'rotation': COLORS_ROTVEC,
    'vec_x': COLORS_ROTVEC,
    'vec_y': COLORS_ROTVEC,
    'vec_z': COLORS_ROTVEC,
    'so2': COLORS_THETA
}


def plot_3d_data(volumes, nrows=5, ncols=3, figsize=(25, 30), cmap='gray'):
    """Plot slices of the volume seen from X, Y and Z.

    Parameters
    ----------
    volume : tensor (torch, numpy)
        map of shape [a,b,c] (I recommend a=b=c).

    Returns
    -------
    None.

    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    v_shape = volumes.shape
    if len(v_shape) != 5:
        volumes = volumes.reshape((v_shape[0], 1)+v_shape[1:])
    for i in range(nrows):
        for j in range(ncols):
            volume = volumes[i][0]
            sum_j = volume.sum(axis=j)
            axes[i, j].imshow(sum_j, cmap=cmap)
            axes[i, j].get_yaxis().set_visible(False)
            axes[i, j].get_xaxis().set_visible(False)
    plt.tight_layout()


def plot_2d_data(images, nrows=5, ncols=2, figsize=(25, 30), cmap='gray'):
    """Plot slices of the volume seen from X, Y and Z.

    Parameters
    ----------
    volume : tensor (torch, numpy)
        map of shape [a,b,c] (I recommend a=b=c).

    Returns
    -------
    None.

    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    i_shape = images.shape
    if len(i_shape) != 4:
        images = images.reshape((i_shape[0], 1)+i_shape[1:])
    for i in range(nrows):
        image1 = images[i][0]
        image2 = images[i+1][0]
        axes[i, 0].imshow(image1, cmap=cmap)
        axes[i, 1].imshow(image2, cmap=cmap)
        axes[i, 0].get_yaxis().set_visible(False)
        axes[i, 1].get_xaxis().set_visible(False)
    plt.tight_layout()


def plot_1d_data(lines, figsize=(25, 30), cmap="gray"):
    fig, axes = plt.subplots(figsize=figsize)
    plt.gcf().subplots_adjust(left=0.125, bottom=0.2, right=1.5,
                              top=0.9, wspace=0.5, hspace=0)
    plt.imshow(lines[:30], cmap='viridis')
    plt.tight_layout()


def show_data(filename, config, nrows=5, ncols=3, figsize=(18, 4),
              cmap='gray'):
    print('Loading %s' % filename)
    dataset = np.load(filename)
    print('Dataset shape:', dataset.shape)
    np.random.shuffle(dataset)
    dataset = torch.Tensor(dataset)
    if config["dimension"] == 1:
        plot_1d_data(dataset)
    elif config["dimension"] == 2:
        d_shape = dataset.shape
        if len(d_shape) == 3:
            dataset = dataset.reshape((d_shape[0], 1)+d_shape[1:])
        _, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize)
        n_samples = nrows * ncols
        for i_img, one_img in enumerate(dataset):
            if i_img > n_samples - 1:
                break
            if len(one_img.shape) == 3:
                one_img = one_img[0]  # channels
                ax = axes[int(i_img // ncols), int(i_img % ncols)]
                ax.imshow(one_img, cmap=cmap)
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
            plt.tight_layout()
    else:
        plot_3d_data(dataset, nrows=nrows, ncols=ncols,
                     figsize=figsize, cmap=cmap)


def get_recon(output, meta_config, liste_points, dataset_path,
              epoch_id=None):
    encoder = train_utils.load_module(
        output, module_name='encoder', epoch_id=epoch_id)
    decoder = train_utils.load_module(
        output, module_name='decoder', epoch_id=epoch_id)
    ckpt = train_utils.load_checkpoint(
        output=output, epoch_id=epoch_id)
    config = ckpt['config']
    dataset = inipip.initialization_dataset(meta_config, dataset_path)
    sub = dataset[liste_points]
    z_nn, _ = encoder(torch.Tensor(sub).to(DEVICE))
    recon, _ = decoder(z_nn)
    recon = np.asarray(recon.cpu().detach().numpy())
    return recon, sub
    try:
        data_dim = functools.reduce(
            (lambda x, y: x * y), encoder.img_shape)
    except AttributeError:
        data_dim = encoder.data_dim

    if recon.shape[-1] == data_dim:
        img_side = int(np.sqrt(data_dim))  # HACK
        recon = recon.reshape(
            (-1,) * len(recon.shape[:-1]) + (img_side, img_side))

    return recon, sub


def show_img_and_recon(output, liste_points, dataset_path,
                       ncols=3, figsize=(18, 20), epoch_id=None, cmap='gray'):
    ckpt = train_utils.load_checkpoint(output=output, epoch_id=epoch_id)
    meta_config = ckpt["meta_config"]
    recon, sub_dataset = get_recon(
        output, meta_config, liste_points, dataset_path, epoch_id=epoch_id)

    if meta_config["dimension"] == 3:
        nrows = len(liste_points)
        volumes = []
        for i in range(nrows):
            volumes.append(recon[i])
            volumes.append(sub_dataset[i])
        volumes = torch.Tensor(volumes)
        plot_3d_data(volumes, nrows=nrows*2, ncols=ncols,
                     figsize=figsize, cmap=cmap)
    elif meta_config["dimension"] == 2:
        nrows = len(liste_points)
        images = []
        for i in range(nrows):
            images.append(recon[i])
            images.append(sub_dataset[i])
        images = torch.Tensor(images)
        plot_2d_data(images, nrows=nrows, ncols=2, figsize=figsize, cmap=cmap)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        axes[0].imshow(recon.reshape(len(liste_points), -1), cmap='viridis')
        print(sub_dataset.shape)
        axes[1].imshow(np.asarray(sub_dataset.view(
            len(liste_points), -1)), cmap='viridis')
        plt.tight_layout()


def plot_variance_explained(output, dataset_path, epoch_id=None, axes=None):
    mus = fvlp.latent_projection(
        output, dataset_path=dataset_path, epoch_id=epoch_id)
    n_pca_components = mus.shape[-1]

    pca, projected_mus = fvlp.pca_projection(mus, n_pca_components)

    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    ax = axes[0]
    ax.plot(
        np.arange(1, n_pca_components+1), pca.explained_variance_ratio_,
        label='Latent dim: %d' % n_pca_components)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('PCA components')
    ax.set_ylabel('Percentage of variance explained')
    ax.set_xticks(np.arange(1, n_pca_components+1, step=1))

    ax = axes[1]
    ax.plot(
        np.arange(1, n_pca_components+1),
        np.cumsum(pca.explained_variance_ratio_),
        label='Latent dim: %d' % n_pca_components)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('PCA components')
    ax.set_ylabel('Cumulative sum of variance explained')
    ax.set_xticks(np.arange(1, n_pca_components+1, step=1))
    return ax


def plot_cryo(ax, output, img_path, labels_path,
              n_pc=2, label_name='focus', epoch_id=None):
    projected_mus, rotations, rotvec, labels, euler, meta_config = fvlp.get_cryo(
        output, img_path, labels_path, n_pca_components=n_pc,
        epoch_id=epoch_id)
    if meta_config["latent_space"] == "so2":
        label_name = "so2"
        for mu, colored_label in zip(projected_mus, euler):
            color_id = int(colored_label)
            im, ax = add_point_plot(n_pc, mu, color_id, ax, label_name)
    elif meta_config["latent_space"] == "rl":
        label_name = "so2"
        for mu, colored_label in zip(projected_mus, euler):
            color_id = int(colored_label)
            im, ax = add_point_plot(n_pc, mu, color_id, ax, label_name)
    elif label_name == "rotation_z1":
        colored_labels = euler.T[0]
        for mu, colored_label in zip(projected_mus, colored_labels):
            color_id = int(colored_label*180/np.pi)
            im, ax = add_point_plot(n_pc, mu, color_id, ax, label_name)
    elif label_name == "rotation_z2":
        colored_labels = euler.T[2]
        for mu, colored_label in zip(projected_mus, colored_labels):
            color_id = int(colored_label*180/np.pi)
            im, ax = add_point_plot(n_pc, mu, color_id, ax, label_name)
    elif label_name == "rotation_y":
        colored_labels = euler.T[1]
        for mu, colored_label in zip(projected_mus, colored_labels):
            color_id = int(colored_label*180/np.pi)
            im, ax = add_point_plot(n_pc, mu, color_id, ax, label_name)
    elif label_name == "vec_x":
        colored_labels = rotvec.T[0]
        for mu, colored_label in zip(projected_mus, colored_labels):
            color_id = int(180*colored_label)
            im, ax = add_point_plot(n_pc, mu, color_id, ax, label_name)
    elif label_name == "vec_y":
        colored_labels = rotvec.T[1]
        for mu, colored_label in zip(projected_mus, colored_labels):
            color_id = int(180*colored_label)
            im, ax = add_point_plot(n_pc, mu, color_id, ax, label_name)
    elif label_name == "vec_z":
        colored_labels = rotvec.T[2]
        for mu, colored_label in zip(projected_mus, colored_labels):
            color_id = int(180*colored_label)
            im, ax = add_point_plot(n_pc, mu, color_id, ax, label_name)

    return im, ax


def add_point_plot(n_pc, mu, color_id, ax, label_name):
    colors = COLORS[label_name]
    if n_pc == 2:
        im = ax.scatter(mu[0], mu[1], c=np.array([colors[color_id]]), s=5)
    else:
        im = ax.scatter(mu[0], mu[1], mu[2],
                        c=np.array([colors[color_id]]))
    return im, ax


def bin_lap_gif(bin_lap_folder, gif_folder, output, path_vae_param, img_path,
                labels_path, epoch_id, label_name, n_pca_components=3,
                n_frames=20):
    if not os.path.exists(bin_lap_folder):
        os.mkdir(bin_lap_folder)
    frames = np.linspace(0, 360, num=n_frames, endpoint=False, dtype='int32')
    gif_name = os.path.join(gif_folder, f'bin_subratio.gif')
    images = []
    projected_mus, labels = fvlp.get_cryo(output, path_vae_param, img_path, labels_path, n_pca_components=n_pca_components,
                                          epoch_id=epoch_id)
    for k in frames:
        print(k)
        filename = os.path.join(bin_lap_folder, f'{k}.png')
        plot_image(n_frames, projected_mus, labels,
                   n_pca_components, k, label_name).savefig(filename)
        images.append(imageio.imread(filename))
        os.remove(filename)
    imageio.mimsave(gif_name, images, fps=10)


def plot_image(n_images, projected_mus, labels, n_pc, angle,
               label_name='focus'):
    colored_labels = labels[label_name]
    if label_name == 'focus':
        colored_labels = [focus for focus in colored_labels]
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')
    for mu, colored_label in zip(projected_mus, colored_labels):
        # if label_name == 'theta' and focus != 2.5:
        #    continue

        if label_name == 'focus':
            color_id = int(10 * colored_label)

        elif label_name in ('theta', 'rotation_x'):
            color_id = int((colored_label + 180))

        elif label_name == 'rotation_y':
            # color_id = int((colored_label + 90))
            color_id = int((colored_label))

        elif label_name in ('quat_x', 'quat_y', 'quat_z', 'quat_w'):
            color_id = int(50*(colored_label+1))

        elif label_name in ('vec_x', 'vec_y', 'vec_z'):
            color_id = int(180*colored_label)

        elif label_name == 'rotation':
            color_id = int(180*colored_label/np.pi)

        elif label_name in ('cosrotationw2', 'cosrotationx2', 'cosrotationy2', 'cosrotationz2'):
            color_id = int(50*(colored_label+1))
        elif label_name == 'sum':
            color_id = int(colored_label)

        else:
            color_id = int(colored_label)

        colors = COLORS[label_name]
        if n_pc == 2:
            im = ax.scatter(mu[0], mu[1], c=np.array([colors[color_id]]), s=5)
        else:
            im = ax.scatter(mu[0], mu[1], mu[2],
                            c=np.array([colors[color_id]]))
    ax.set_title(label_name)
    ax.view_init(30, angle)
    plt.close()
    return fig
