"""Tools to analyze the results of vae learning and latent space computing."""
import glob
import importlib
import neural_network as nn1
import numpy as np
import pandas as pd
import torch
import train_utils
import visualization
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from ray.tune.analysis import Analysis
import initialization_pipeline as inipip

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

N_PCA_COMPONENTS = 5


def reload_libs():
    import from_vae_to_latent_space

    importlib.reload(from_vae_to_latent_space)
    importlib.reload(inipip)
    importlib.reload(visualization)
    importlib.reload(nn1)
    importlib.reload(train_utils)


def get_all_logdirs(main_dir, select_dict={}):

    all_logdirs = []
    analysis = Analysis(main_dir)

    all_dataframes = analysis.trial_dataframes

    for logdir, train_dict in analysis.get_all_configs().items():

        keep_logdir = True
        for param_name, param_value in select_dict.items():
            if param_name not in train_dict.keys():
                keep_logdir = False
                break
            if train_dict[param_name] != param_value:
                keep_logdir = False
                break

        if not keep_logdir:
            continue

        if logdir not in all_dataframes.keys():
            # This means this trial errored
            continue

        all_logdirs.append(logdir)
    print('Found %d logdirs.' % len(all_logdirs))

    return all_logdirs


def get_best_logdir(main_dir, select_dict={}, metric='average_loss'):
    analysis = Analysis(main_dir)

    all_dataframes = analysis.trial_dataframes

    best_logdir = 'none'
    min_metric_value = 1e10
    for logdir, train_dict in analysis.get_all_configs().items():
        keep_logdir = True
        for param_name, param_value in select_dict.items():
            if param_name not in train_dict.keys():
                keep_logdir = False
                break
            if train_dict[param_name] != param_value:
                keep_logdir = False
                break

        if not keep_logdir:
            continue

        if logdir not in all_dataframes.keys():
            # This means this trial errored
            continue
        metric_value = all_dataframes[logdir][metric].iloc[-1]
        if metric_value < min_metric_value:
            min_metric_value = metric_value
            best_logdir = logdir

    print("Best logdir (%s) with required parameters is" % metric, best_logdir)
    print(select_dict)
    return best_logdir


def dif_reconstruction(output, epoch_id=None):
    ckpt = train_utils.load_checkpoint(
        output=output, epoch_id=epoch_id)['val_losses'][-1]['reconstruction']
    return ckpt


def get_best_logdir1(main_dir, select_dict={}, metric='average_loss'):
    analysis = Analysis(main_dir)

    all_dataframes = analysis.trial_dataframes

    best_logdir = 'none'
    min_metric_value = 1e10
    if metric == 'average_loss':
        for logdir, train_dict in analysis.get_all_configs().items():
            keep_logdir = True
            for param_name, param_value in select_dict.items():
                if param_name not in train_dict.keys():
                    keep_logdir = False
                    break
                if train_dict[param_name] != param_value:
                    keep_logdir = False
                    break

            if not keep_logdir:
                continue

            if logdir not in all_dataframes.keys():
                # This means this trial errored
                continue
            metric_value = all_dataframes[logdir][metric].iloc[-1]
            if metric_value < min_metric_value:
                min_metric_value = metric_value
                best_logdir = logdir
    else:
        for logdir, train_dict in analysis.get_all_configs().items():
            ckpts = glob.glob(
                '%s/checkpoint_*/epoch_*_checkpoint.pth' % logdir)
            if len(ckpts) != 0:
                metric_value = dif_reconstruction(logdir, epoch_id=None)
                print(metric_value)
                if metric_value < min_metric_value:
                    min_metric_value = metric_value
                    best_logdir = logdir
    print("Best logdir (%s) with required parameters is" % metric, best_logdir)
    print(select_dict)
    return best_logdir


def pca_projection(mus, n_pca_components=N_PCA_COMPONENTS):
    pca = PCA(n_components=n_pca_components)
    pca.fit(mus)
    projected_mus = pca.transform(mus)
    return pca, projected_mus


def latent_projection(output,  dataset_path, epoch_id=None):
    ckpt = train_utils.load_checkpoint(output=output, epoch_id=epoch_id)
    config = ckpt["config"]
    meta_config = ckpt["meta_config"]
    dataset = inipip.initialization_dataset(meta_config, dataset_path)
    encoder = train_utils.load_module(
        output, module_name="encoder", epoch_id=epoch_id
    )
    z_nn, matrix = encoder(dataset)
    return z_nn.detach().numpy()


def vae_matrix(output, path_vae_param, dataset_path, epoch_id=None):
    ckpt = train_utils.load_checkpoint(output=output, epoch_id=epoch_id)
    config = ckpt["config"]
    meta_config = ckpt["meta_config"]
    dataset = inipip.initialization_dataset(meta_config, dataset_path)
    encoder = train_utils.load_module(
        output, path_vae_param, module_name="encoder", epoch_id=epoch_id
    )
    z_nn, matrix = encoder(dataset)
    return z_nn.detach().numpy()


def get_sigma(output, dataset_path, epoch_id=None):
    ckpt = train_utils.load_checkpoint(output=output, epoch_id=epoch_id)
    config = ckpt["config"]
    meta_config = ckpt["meta_config"]
    dataset = inipip.initialization_dataset(meta_config, dataset_path)
    encoder = train_utils.load_module(
        output, module_name="encoder", epoch_id=epoch_id
    )
    z_nn, matrix = encoder(dataset)
    return z_nn.detach().numpy()


def from_image_to_image(output, dataset_path, meta_config, n_img, epoch_id=None):
    ckpt = train_utils.load_checkpoint(output=output, epoch_id=epoch_id)
    config = ckpt['config']
    encoder = train_utils.load_module(
        output, module_name="encoder", epoch_id=epoch_id
    )
    decoder = train_utils.load_module(
        output, module_name="decoder", epoch_id=epoch_id
    )
    dataset = inipip.initialization_dataset(meta_config, dataset_path)
    dataset = dataset[:20]
    z_nn, matrix = encoder(dataset)
    recon, scale_b = decoder(z_nn)
    return recon


def get_cryo_labels(labels_path=None, from_id=1, to_id=None):
    if labels_path is not None:
        rotations, rotvec, labels, euler = inipip.open_labels(
            labels_path)
    return rotations, rotvec, labels, euler


def get_cryo(output, dataset_path, labels_path=None, n_pca_components=2,
             epoch_id=None):
    rotations, rotvec, labels, euler = get_cryo_labels(labels_path)
    ckpt = train_utils.load_checkpoint(output=output, epoch_id=epoch_id)
    meta_config = ckpt['config']
    encoder = train_utils.load_module(
        output, module_name="encoder", epoch_id=epoch_id
    )
    dataset = inipip.initialization_dataset(meta_config, dataset_path)
    z_nn, matrix = encoder(dataset)
    # z_nn = latent_projection(output, dataset_path, epoch_id)
    # _, projected_mus = pca_projection(
    #     mus=z_nn, n_pca_components=n_pca_components)

    return z_nn.detach().numpy(), rotations, rotvec, labels, euler, ckpt['meta_config']


def f(x):
    if x == 0:
        return -1
    else:
        return 1


def rot_matrix(labels_path):
    Coords = pd.read_csv(labels_path)
    real_matrix = pd.DataFrame(
        Coords.apply(
            lambda row: R.from_quat(
                [row["quat_x"], row["quat_y"], row["quat_z"], row["quat_w"]]
            ).as_matrix(),
            axis=1,
        )
    )
    return real_matrix


def prediction_rotation(output, path_vae_param, dataset_path, labels_path):
    ckpt = train_utils.load_checkpoint(output=output)
    config = ckpt["nn_architecture"]
    latent_dim = config["latent_dim"]
    real_matrix, _, _, _ = inipip.open_labels(labels_path)
    if config["latent_space_definition"] == 0:
        vae_matrices = latent_projection(output, path_vae_param, dataset_path)
        n = len(vae_matrices)
        vae_matrix1 = []
        for i in range(n):
            vae_matrix1.append(R.from_rotvec(vae_matrices[i]).as_matrix())

    else:
        mus, z, matrix = vae_matrix(output, path_vae_param, dataset_path)
        matrix_from_mus = nn1.rot_mat_tensor(mus, 3)
        with_mus = compute_ground_truth(real_matrix, matrix_from_mus)
        if latent_dim == 6:
            with_matrix = compute_ground_truth(real_matrix, matrix)
        else:
            with_matrix = with_mus
    return with_mus, with_matrix


def compute_ground_truth(real_matrix, estimate_matrix):
    r0_matrices = []
    dist_0 = np.linalg.norm(np.asarray(real_matrix) -
                            np.asarray(estimate_matrix))
    dist_0 = []
    n = len(estimate_matrix)
    for i in range(n):
        r0_matrices.append(np.asarray(
            estimate_matrix[i]).dot(real_matrix[i].T))
        dist_0.append(np.linalg.norm(np.asarray(real_matrix[i]) -
                                     np.asarray(estimate_matrix[i])))
    dist_0 = pd.DataFrame(dist_0.copy())
    r0_rotvecs = []
    for i in range(n):
        r0_rotvecs.append(R.from_matrix(r0_matrices[i]).as_rotvec())
    new_rot = []
    for i in range(n):
        rot = np.linalg.norm(r0_rotvecs[i])
        new_rot.append([rot, r0_rotvecs[i] / rot])
    new_rot = pd.DataFrame(new_rot)
    r0_rotvecs_df = pd.DataFrame(new_rot[1][0]).T
    r0_rotvecs_df .columns = ["vec_x", "vec_y", "vec_z"]
    for i in range(1, n):
        new1 = pd.DataFrame(new_rot[1][i]).T
        new1.columns = ["vec_x", "vec_y", "vec_z"]
        r0_rotvecs_df = r0_rotvecs_df.append(new1.copy(), ignore_index=True)
    r0_rotvecs_df["rotation"] = new_rot[0]

    return r0_matrices, r0_rotvecs, r0_rotvecs_df, dist_0


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
    labels1 = np.load(path, allow_pickle=True).T[0].T
    n = labels1.shape[0]
    labels = np.zeros((n, 4))
    for i in range(n):
        labels[i] = labels1[i]
    rot_representation = R.from_quat(labels)
    rotation = rot_representation.as_matrix()
    rotvec = rot_representation.as_rotvec()
    euler = rot_representation.as_euler("ZYZ")
    return rotation, rotvec, labels, euler


def analyze_sigma_distribution(output, path_vae_param, dataset_path, labels_path):
    ckpt = train_utils.load_checkpoint(output=output)
    config = ckpt["nn_architecture"]
    latent_dim = config["latent_dim"]
    real_matrix, _, _, _ = inipip.open_labels(labels_path)
