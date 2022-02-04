"""Tools to analyze the latent space."""

import glob
from ai import cs
from scipy.spatial.transform import Rotation as R
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
import pandas as pd
import csv
import importlib
import os
import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from geomstats.geometry.discrete_curves import DiscreteCurves
from ray.tune.analysis import Analysis
import geom_utils
import train_utils
import initialization_pipeline as inipip
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

N_PCA_COMPONENTS = 5

TOY_LOGVARX_TRUE = [-10, -5, -3.22, -2, -1.02, -0.45, 0]
TOY_STD_TRUE = np.sqrt(np.exp(TOY_LOGVARX_TRUE))
TOY_N = [10000, 100000]


def reload_libs():
    import analyze
    importlib.reload(analyze)
    import cryo_dataset
    importlib.reload(cryo_dataset)
    import vis
    importlib.reload(vis)
    import toylosses
    importlib.reload(toylosses)
    import nn
    importlib.reload(nn)
    import train_utils
    importlib.reload(train_utils)
    import visualization
    importlib.reload(visualization)


def latent_projection(output, path_vae_param, dataset_path, meta_config, algo_name='vae',
                      epoch_id=None):
    ckpt = train_utils.load_checkpoint(
        output=output, epoch_id=epoch_id)
    config = ckpt['nn_architecture']
    if "is_3d" not in config.keys():
        config["is_3d"] = True
    dataset = inipip.initialization_dataset(meta_config, dataset_path)
    encoder = train_utils.load_module(
        output, path_vae_param, module_name='encoder', epoch_id=epoch_id)
    mus, logvar, matrix = encoder(dataset)
    return matrix.detach().numpy()


def latent_projection(output, path_vae_param, dataset_path, algo_name='vae',
                      epoch_id=None):
    ckpt = train_utils.load_checkpoint(
        output=output, epoch_id=epoch_id)
    dataset = np.load(dataset_path)

    if 'spd_feature' in ckpt['nn_architecture']:
        spd_feature = ckpt['nn_architecture']['spd_feature']
        if spd_feature is not None:
            dataset = train_utils.spd_feature_from_matrix(
                dataset, spd_feature=spd_feature)

    encoder = train_utils.load_module(
        output, path_vae_param, module_name='encoder', epoch_id=epoch_id)
    dataset = torch.Tensor(dataset)
    dataset = torch.utils.data.TensorDataset(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)

    mus = []
    for i, data in enumerate(loader):
        data = data[0].to(DEVICE)  # extract from loader's list
        if len(data.shape) == 3:
            data = torch.unsqueeze(data, dim=0)
        assert len(data.shape) == 4
        mu, logvar = encoder(data)
        mus.append(np.array(mu.cpu().detach()))

    mus = np.array(mus).squeeze()
    return mus


def pca_projection(mus, n_pca_components=N_PCA_COMPONENTS):
    pca = PCA(n_components=n_pca_components)
    pca.fit(mus)
    projected_mus = pca.transform(mus)
    return pca, projected_mus


def plot_kde(ax, projected_mus):
    x = projected_mus[:, 0]
    y = projected_mus[:, 1]
    data = np.vstack([x, y])
    kde = gaussian_kde(data)

    # Evaluate on a regular grid
    xgrid = np.linspace(-4, 4, 200)
    ygrid = np.linspace(-5, 5, 200)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    # Plot the result as an image
    ax.imshow(Z.reshape(Xgrid.shape),
              origin='lower', aspect='auto',
              extent=[-4, 4, -5, 5],
              cmap='Blues')
    return ax


def get_subset_fmri(output, path_vae_param, metadata_csv, ses_ids=None, task_names=None,
                    epoch_id=None, n_pcs=2):
    paths_subset = []

    tasks_subset = []
    ses_subset = []
    times_subset = []

    with open(metadata_csv, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            path, ses, task, run, time = row
            if path == 'path':
                continue
            ses = int(ses)
            run = int(run)
            time = int(time)
            if (task_names is not None) and (task not in task_names):
                continue
            if run != 1:
                # Only take one run per session
                continue
            if (ses_ids is not None) and (ses not in ses_ids):
                continue

            paths_subset.append(path)

            tasks_subset.append(task)
            ses_subset.append(ses)
            times_subset.append(time)

    subset = [np.load(one_path) for one_path in paths_subset]
    subset = np.array(subset)

    # Note: dataset needs to be unshuffled here
    mus = latent_projection(output, path_vae_param, subset, epoch_id=epoch_id)
    _, projected_mus = pca_projection(mus, n_pcs)

    labels_subset = {
        'task': tasks_subset, 'ses': ses_subset, 'time': times_subset}

    return projected_mus, labels_subset


def load_losses(output, epoch_id=None,
                crit_name='neg_elbo', mode='train'):
    ckpt = train_utils.load_checkpoint(
        output=output, epoch_id=epoch_id)

    losses = ckpt['%s_losses' % mode]
    losses = [loss[crit_name] for loss in losses]

    return losses


def submanifold_from_t_and_decoder_in_euclidean(
        t, decoder, logvarx=1, with_noise=False):
    """
    Generate data using generative model from decoder.
    Euclidean Gaussian noise uses the logvarx given as input.
    Logvarx is fixed, as opposed to using logvarx generated
    from z by decoder.
    """
    t = torch.Tensor(t).to(DEVICE)
    mux, _ = decoder(t)
    mux = mux.cpu().detach().numpy()
    n_samples, data_dim = mux.shape

    generated_x = mux
    if with_noise:
        generated_x = np.zeros((n_samples, data_dim))
        for i in range(n_samples):
            logvar = logvarx
            sigma = np.sqrt(np.exp((logvar)))
            eps = np.random.normal(
                loc=0, scale=sigma, size=(1, data_dim))
            generated_x[i] = mux[i] + eps

    return mux, generated_x


def submanifold_from_t_and_decoder_on_manifold(
        t, decoder, logvarx=1, manifold_name='h2', with_noise=False):
    """
    The decoder generate on the tangent space of a manifold.
    We use Exp to bring these points on the manifold.
    We add a Gaussian noise at each point.
    To this aim, we use a wrapped Gaussian: we generate a Gaussian noise
    at the tangent space of the point, and use the Exp at the point to
    get a point on the manifold.
    Logvarx is fixed, as opposed to using logvarx generated
    from z by decoder.
    Extrinsic representation for points on manifold (3D).
    """
    t = torch.Tensor(t).to(DEVICE)
    mux, _ = decoder(t)
    mux = mux.cpu().detach().numpy()
    n_samples, data_dim = mux.shape

    mux = geom_utils.convert_to_tangent_space(mux, manifold_name=manifold_name)
    manifold, base_point = geom_utils.manifold_and_base_point(manifold_name)

    mux_riem = manifold.metric.exp(mux, base_point=base_point)

    generated_x = mux_riem
    if with_noise:
        scale = np.sqrt(np.exp(logvarx))
        eps = np.random.normal(
            loc=0, scale=scale, size=(n_samples, data_dim+1))  # HACK!
        eps = manifold.projection_to_tangent_space(
            vector=eps, base_point=mux_riem)

        generated_x = manifold.metric.exp(eps, base_point=mux_riem)

    return mux_riem, generated_x


def submanifold_from_t_and_decoder_on_tangent_space(
        t, decoder, logvarx=1, manifold_name='h2', with_noise=False):
    """
    Bring the generated points back on the tangent space
    at the chosen basepoint and uses 2D..
    Logvarx is fixed, as opposed to using logvarx generated
    from z by decoder.
    Intrinsic representation for vectors on tangent space (2D).
    """
    t = torch.Tensor(t).to(DEVICE)
    mux_riem, generated_x = submanifold_from_t_and_decoder_on_manifold(
        t, decoder, logvarx, manifold_name)

    manifold, base_point = geom_utils.manifold_and_base_point(manifold_name)

    mux_riem_on_tangent_space = manifold.metric.log(
        mux_riem, base_point=base_point)
    if manifold_name == 's2':
        mux_riem_on_tangent_space = mux_riem_on_tangent_space[:, :2]
    elif manifold_name == 'h2':
        mux_riem_on_tangent_space = mux_riem_on_tangent_space[:, 1:]

    generated_x_on_tangent_space = mux_riem_on_tangent_space
    if with_noise:
        generated_x_on_tangent_space = manifold.metric.log(
            generated_x, base_point=base_point)
        if manifold_name == 's2':
            generated_x_on_tangent_space = generated_x_on_tangent_space[:, :2]
        elif manifold_name == 'h2':
            generated_x_on_tangent_space = generated_x_on_tangent_space[:, 1:]
    return mux_riem_on_tangent_space, generated_x_on_tangent_space


def true_submanifold_from_t_and_decoder(
        t, decoder, manifold_name='r2',
        with_noise=False, logvarx_true=None):
    """
    Logvarx is fixed, as opposed to using logvarx generated
    from z by decoder.
    """
    if manifold_name == 'r2':
        x_novarx, x = submanifold_from_t_and_decoder_in_euclidean(
            t, decoder, logvarx=logvarx_true, with_noise=with_noise)

    elif manifold_name in ['s2', 'h2']:
        x_novarx, x = submanifold_from_t_and_decoder_on_manifold(
            t, decoder, logvarx=logvarx_true,
            manifold_name=manifold_name, with_noise=with_noise)
    else:
        raise ValueError('Manifold not supported.')

    return x_novarx, x


def learned_submanifold_from_t_and_decoder(
        t, decoder, vae_type='gvae_tgt',
        manifold_name='r2', with_noise=False):
    """
    Logvarx is fixed to be the true logvarx,
    as opposed to using logvarx generated
    from z by decoder.
    """
    if manifold_name in ['r2', 'r3']:
        x_novarx, x = submanifold_from_t_and_decoder_in_euclidean(
            t, decoder, logvarx=decoder.logvarx_true, with_noise=with_noise)

    elif manifold_name in ['s2', 'h2']:
        x_novarx, x = submanifold_from_t_and_decoder_on_manifold(
            t, decoder, logvarx=decoder.logvarx_true,
            manifold_name=manifold_name, with_noise=with_noise)
    else:
        raise ValueError('Manifold not supported.')

    return x_novarx, x


def true_submanifold_from_t_and_output(
        t, output, algo_name='vae', manifold_name='r2',
        epoch_id=None, with_noise=False):
    """
    Generate:
    - true_x_no_var: true submanifold used in the experiment output
    - true_x: data generated from true model
    """
    decoder_true_path = '%s/decoder_true.pth' % output
    decoder_true = torch.load(decoder_true_path, map_location=DEVICE)

    logvarx_true = None
    if with_noise:
        ckpt = train_utils.load_checkpoint(output)
        logvarx_true = ckpt['nn_architecture']['logvarx_true']

    true_x_novarx, true_x = true_submanifold_from_t_and_decoder(
        t, decoder_true, manifold_name=manifold_name,
        with_noise=with_noise, logvarx_true=logvarx_true)
    return true_x_novarx, true_x


def learned_submanifold_from_t_and_output(
        t, output, algo_name='vae', manifold_name='r2',
        epoch_id=None, with_noise=False):
    """
    Generate:
    - true_x_no_var: true submanifold used in the experiment output
    - true_x: data generated from true model
    """
    decoder = train_utils.load_module(
        output, module_name='decoder', epoch_id=epoch_id)

    # logvarx_true = None
    # if with_noise:
    #    ckpt = train_utils.load_checkpoint(
    #        output, algo_name=algo_name, epoch_id=epoch_id)
    #    logvarx_true = ckpt['nn_architecture']['logvarx_true']
    #    # TODO(nina): Decide if using the truth or cst -5

    x_novarx, x = learned_submanifold_from_t_and_decoder(
        t, decoder, manifold_name=manifold_name,
        with_noise=with_noise)
    return x_novarx, x


def learned_submanifold_from_t_and_vae_type(
        t, vae_type, logvarx_true, n,
        algo_name='vae', manifold_name='r2',
        epoch_id=None, with_noise=False,
        main_dir='/ray_results/Train/'):
    """
    Generate:
    - true_x_no_var: true submanifold used in the experiment output
    - true_x: data generated from true model
    """
    train_dict = {}
    train_dict['algo_name'] = algo_name
    train_dict['manifold_name'] = manifold_name
    train_dict['vae_type'] = vae_type
    train_dict['logvarx_true'] = logvarx_true
    train_dict['n'] = n

    if vae_type in ['gvae', 'gvae_tgt']:
        output = get_last_logdir(
            select_dict=train_dict, main_dir=main_dir)

        x_novarx, x = learned_submanifold_from_t_and_output(
            t, output=output,
            algo_name=algo_name, manifold_name=manifold_name,
            epoch_id=epoch_id, with_noise=with_noise)
    elif vae_type == 'vae':
        output = get_last_logdir(
            select_dict=train_dict, main_dir=main_dir)

        x_novarx, x = learned_submanifold_from_t_and_output(
            t, output=output,
            algo_name=algo_name, manifold_name='r3',
            epoch_id=epoch_id, with_noise=with_noise)
    elif vae_type == 'vae_proj':
        train_dict['vae_type'] = 'vae'
        output = get_last_logdir(
            select_dict=train_dict, main_dir=main_dir)

        x_novarx, x = learned_submanifold_from_t_and_output(
            t, output=output,
            algo_name=algo_name, manifold_name='r3',
            epoch_id=epoch_id, with_noise=with_noise)

        norms = np.linalg.norm(x_novarx, axis=1)
        norms = np.expand_dims(norms, axis=1)
        x_novarx = x_novarx / norms

        norms = np.linalg.norm(x, axis=1)
        norms = np.expand_dims(norms, axis=1)
        x = x / norms
    elif vae_type == 'pga':
        train_dict['vae_type'] = 'gvae_tgt'
        output = get_last_logdir(
            select_dict=train_dict, main_dir=main_dir)

        synthetic_dataset_in_tgt = np.load(os.path.join(
            output, 'synthetic/dataset.npy'))
        pca = PCA(n_components=1)
        pca.fit(synthetic_dataset_in_tgt)

        component_extrinsic = geom_utils.convert_to_tangent_space(
            pca.components_, manifold_name='s2')
        manifold, base_point = geom_utils.manifold_and_base_point(
            manifold_name)
        geodesic = manifold.metric.geodesic(
            initial_point=base_point, initial_tangent_vec=component_extrinsic)

        x_novarx = geodesic(t)
        x = x_novarx

    return x_novarx, x


def toyoutput_dir(manifold_name, logvarx_true, n, vae_type='gvae'):
    main_dir = '/scratch/users/nmiolane/toyoutput_manifold_%s' % vae_type
    output = os.path.join(main_dir, 'logvarx_%s_n_%d_%s' % (
        logvarx_true, n, manifold_name))
    return output


def parse_train_dir(train_dir):
    train_dir = train_dir.replace('=', '_').replace(',', '_')
    train_dict = {}
    for param in ['_algo_name_', '_manifold_name_', '_vae_type_']:
        substring = train_dir[train_dir.find(param)+len(param):]
        param_value = substring.split('_')[0]
        train_dict[param[1:-1]] = param_value
    for param in ['_logvarx_true_', '_n_']:
        substring = train_dir[train_dir.find(param)+len(param):]
        param_value = substring.split('_')[0]
    return train_dict


def ray_output_dir(train_dict, main_dir='/ray_results/Train'):
    output_dirs = []
    for _, train_dirs, _ in os.walk(main_dir):
        for train_dir in train_dirs:
            keep_dir = True
            for param_name, param_value in train_dict.items():
                pattern = param_name + '=' + str(param_value)
                if pattern not in train_dir:
                    keep_dir = False
                    break
            if not keep_dir:
                continue
            output_dirs.append(os.path.join(main_dir, train_dir))
    if len(output_dirs) > 1:
        print('Found more than 1 dir, returning last one.')
    return output_dirs[-1]


def squared_dist_between_submanifolds(manifold_name,
                                      vae_type='gvae_tgt',
                                      all_logvarx_true=TOY_LOGVARX_TRUE,
                                      all_n=TOY_N,
                                      epoch_id=100,
                                      n_samples=1000,
                                      extrinsic_or_intrinsic='extrinsic'):
    """
    Compute:
    d(N1, N2) = int d^2(f_theta1(z), f_theta2(z)) dmu(z)
    by Monte-Carlo approximation,
    when:
    - mu is the standard normal on the 1D latent space,
    - d is the extrinsic (r3) or intrinsic dist on manifold_name
    """
    dists = np.zeros((len(all_n), len(all_logvarx_true)))

    t = np.random.normal(size=(n_samples,))

    for i_logvarx_true, logvarx_true in enumerate(all_logvarx_true):
        for i_n, n in enumerate(all_n):
            output_decoder_true = toyoutput_dir(
                vae_type='gvae_tgt', manifold_name=manifold_name,
                logvarx_true=logvarx_true, n=n)
            submanifold_true, _ = true_submanifold_from_t_and_output(
                t, output=output_decoder_true, manifold_name=manifold_name)

            submanifold_learned, _ = learned_submanifold_from_t_and_vae_type(
                t=t, vae_type=vae_type,
                logvarx_true=logvarx_true, n=n,
                manifold_name=manifold_name, epoch_id=epoch_id)

            if extrinsic_or_intrinsic == 'intrinsic':
                curves_space = DiscreteCurves(
                    ambient_manifold=geom_utils.MANIFOLD[manifold_name])
                curves_space_metric = curves_space.l2_metric

                dist = curves_space_metric.dist(
                    submanifold_true, submanifold_learned) ** 2
            else:
                dist = np.linalg.norm(
                    submanifold_learned - submanifold_true) ** 2

            dists[i_n, i_logvarx_true] = dist / n_samples
    return dists


def make_gauss_hist(n_bins, m=0, s=1):
    x_bin_min = m - 3 * s
    x_bin_max = m + 3 * s
    x = np.linspace(x_bin_min, x_bin_max, n_bins, dtype=np.float64)
    h = np.exp(-(x - m)**2 / (2 * s**2))
    return x, h / h.sum()


def squared_w2_between_submanifolds(manifold_name,
                                    vae_type,
                                    all_logvarx_true,
                                    all_n,
                                    extrinsic_or_intrinsic='extrinsic',
                                    n_bins=5,
                                    sinkhorn=False,
                                    main_dir='/ray_results/Train'):
    manifold, base_point = geom_utils.manifold_and_base_point(
        manifold_name)

    w2_dists = np.zeros((len(all_n), len(all_logvarx_true)))

    x_bins_a, a = make_gauss_hist(n_bins, m=0, s=1)
    x_bins_b, b = make_gauss_hist(n_bins, m=0, s=1)
    assert np.all(x_bins_a == x_bins_b)
    x = x_bins_a
    train_dict = {}
    train_dict['vae_type'] = vae_type
    train_dict['manifold_name'] = manifold_name

    for i_logvarx_true, logvarx_true in enumerate(all_logvarx_true):
        for i_n, n in enumerate(all_n):
            train_dict['n'] = n
            train_dict['logvarx_true'] = logvarx_true
            train_dict['vae_type'] = 'gvae_tgt'

            output_decoder_true = get_last_logdir(
                select_dict=train_dict, main_dir=main_dir)

            M2 = np.zeros((n_bins, n_bins))
            for i in range(n_bins):
                for j in range(n_bins):
                    zi = np.expand_dims(np.expand_dims(x[i], axis=0), axis=1)
                    xi, _ = true_submanifold_from_t_and_output(
                        t=zi, output=output_decoder_true,
                        manifold_name=manifold_name)

                    zj = np.expand_dims(np.expand_dims(x[j], axis=0), axis=1)
                    xj, _ = learned_submanifold_from_t_and_vae_type(
                        t=zj, manifold_name=manifold_name, vae_type=vae_type,
                        logvarx_true=logvarx_true, n=n, main_dir=main_dir)

                    if extrinsic_or_intrinsic == 'intrinsic':
                        sq_dist = manifold.metric.squared_dist(xi, xj)
                    else:
                        sq_dist = np.linalg.norm(xj - xi) ** 2
                    M2[i, j] = sq_dist

            if sinkhorn:
                d_emd2 = ot.sinkhorn2(a, b, M2, 1e-3)
            else:
                d_emd2 = ot.emd2(a, b, M2)

            w2_dists[i_n, i_logvarx_true] = d_emd2
    return w2_dists


def get_cryo_labels(labels_path=None, from_id=1, to_id=None):
    labels = {}
    labels['focus'] = []
    labels['theta'] = []
    if labels_path is not None:
        labels = pd.read_csv(labels_path)
        for column in labels.columns:
            labels[column] = labels[column].astype(float)
        # labels['rotation_x'] = labels['rotation_x'].astype(float)
        # labels['rotation_y'] = labels['rotation_y'].astype(float)
        # labels['rotation_z'] = labels['rotation_z'].astype(float)
        # labels['focus'] = labels['focus'].astype(float)
    return labels


def get_cryo(output, path_vae_param, dataset_path,
             labels_path=None, n_pca_components=2, epoch_id=None):

    labels = get_cryo_labels(labels_path)
    mus = latent_projection(
        output, path_vae_param, dataset_path, epoch_id)
    print(mus.shape)
    _, projected_mus = pca_projection(
        mus=mus, n_pca_components=n_pca_components)

    return projected_mus, labels


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


def get_last_logdir(main_dir, select_dict={}):
    all_logdirs = get_all_logdirs(main_dir, select_dict)
    if len(all_logdirs) > 0:
        last_logdir = all_logdirs[-1]
    else:
        last_logdir = None

    print("Last logdir with required parameters is", last_logdir)
    print(select_dict)
    return last_logdir


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


def crit_between_manifolds(train_dict,
                           all_logvarx_true,
                           all_n,
                           crit_name='neg_elbo',
                           main_dir=''):
    """
    Returns a table of the crit, where:
    - Lines are different values of n
    - Columns are different values of logvarx_true
    """
    crit = np.zeros((len(all_n), len(all_logvarx_true)))

    for i_logvarx_true, logvarx_true in enumerate(all_logvarx_true):
        for i_n, n in enumerate(all_n):
            train_dict['logvarx_true'] = logvarx_true
            train_dict['n'] = n
            output = get_last_logdir(
                select_dict=train_dict, main_dir=main_dir)

            val_losses = load_losses(
                output, 'vae', crit_name=crit_name,
                epoch_id=int(100), mode='val')
            crit[i_n, i_logvarx_true] = val_losses[100]
    return crit


def reconstruction(output, z, algo_name='vae', epoch_id=None):
    decoder = train_utils.load_module(output,
                                      module_name='decoder', epoch_id=epoch_id)
    recon, _ = decoder(z)
    recon = recon.cpu().detach().numpy()
    return recon


SO3 = SpecialOrthogonal(n=3, point_type="vector")


def convert_quat_euler(quaternion):
    i = 0
    while '  ' in quaternion:
        quaternion = quaternion.replace('  ', ' ')
    quaternion = quaternion.replace(' ]', ']')
    quaternion = quaternion.replace('[ ', '[')
    quaternion = quaternion[1:-1].split(' ')
    quaternion = list(map(float, quaternion))
    r = R.from_rotvec(SO3.rotation_vector_from_quaternion(quaternion))
    rotation = r.as_euler('zyx', degrees=True)
    return rotation


def quat(quaternion):
    i = 0
    while '  ' in quaternion:
        i += 1
        quaternion = quaternion.replace('  ', ' ')
    quaternion = quaternion.replace(' ]', ']')
    quaternion = quaternion.replace('[ ', '[')
    quaternion = quaternion[1:-1].split(' ')
    quaternion = list(map(float, quaternion))
    return quaternion


def convert_quat_rotvec(quaternion):
    i = 0
    while '  ' in quaternion:
        i += 1
        quaternion = quaternion.replace('  ', ' ')
    quaternion = quaternion.replace(' ]', ']')
    quaternion = quaternion.replace('[ ', '[')
    quaternion = quaternion[1:-1].split(' ')
    quaternion = list(map(float, quaternion))
    r = R.from_quat(quaternion)
    vect = r.as_rotvec()
    rot = np.sqrt(vect[0]**2+vect[1]**2+vect[2]**2)
    # rotation = SO3.rotation_vector_from_quaternion(quaternion)
    return np.asarray([vect[0], vect[1], vect[2], rot])


def select_representation(SUB_PATH, DATASET, LABELS, N_SUB, sub_dataset_description={'representation': 'quaternion'}):
    dataset = DATASET[:N_SUB]
    representation = sub_dataset_description['representation']
    if representation == 'quaternion':
        labels = LABELS[:N_SUB][[
            'focus', 'quat_x', 'quat_y', 'quat_z', 'quat_w']]
    if representation == 'euler':
        labels = LABELS[:N_SUB][[
            'focus', 'rotation_x', 'rotation_y', 'rotation_z', 'rotation_x_mod']]
    if representation == 'axis-angle':
        labels = LABELS[:N_SUB][[
            'focus', 'vec_x', 'vec_y', 'vec_z', 'rotation']]
    paths = sub_dataset_description['paths']
    conditions = sub_dataset_description['conditions']
    for key in conditions:
        labels = labels[labels[key] >= conditions[key][0]]
        labels = labels[labels[key] <= conditions[key][1]]
    dataset = dataset[labels.index]
    np.save(SUB_PATH+paths[0], dataset)
    labels.to_csv(SUB_PATH+paths[1])
    np.save(SUB_PATH+paths[0], dataset)
    labels.to_csv(SUB_PATH+paths[1])


def f(x):
    if x == 0:
        return -1
    else:
        return 1


def prediction_rotation(output, dataset_path, labels_path):
    real_matrix = rot_matrix(labels_path)
    real_matrix.columns = ['RMatrix']
    real_matrix[['R_0Euler_x', 'R_0Euler_y', 'R_0Euler_z']] = real_matrix.apply(
        lambda row: R.from_matrix(row['RMatrix']).as_euler('xyz', degrees=True), axis=1).tolist()
    projected_mus, labels = get_cryo(output, dataset_path, n_pca_components=3)
    # Y est le rayon
    mus = pd.DataFrame(projected_mus)
    mus.columns = ['x', 'y', 'z']
    Coords = mus.apply(lambda row: cs.cart2sp(
        x=row['x'], y=row['y'], z=row['z']), axis=1)
    Coords = pd.DataFrame(Coords)
    Coords[['r', 'theta', 'phi']] = Coords[0].tolist()  # a renommer
    max_y = max(Coords['r'])
    b = 90
    a = b/max_y
    Coords['y1'] = -90+a*Coords['r']
    Coords['y2'] = 90-a*Coords['r']
    Test = pd.DataFrame()
    Test['intermediaire'] = real_matrix['R_0Euler_y']
    Test['r'] = Coords['r']
    Test['y'] = Test.apply(lambda row: row['r']*a *
                           f(int(1+(row['intermediaire']/90))), axis=1)
    Coords['y'] = Test['y']
    Coords['z'] = Coords.apply(
        lambda row: row['phi']/2 - (1-int(row['theta']/np.pi))*180, axis=1)
    # Coords['y'] = a*Coords['y']**2+b*Coords['y']+c
    del Coords[0]
    Coords['x'] = Coords['theta']
    predicted_matrix1 = pd.DataFrame(Coords.apply(lambda row: R.from_euler(
        'zyx', [row['z'], row['y1'], row['x']]).as_matrix(), axis=1))
    predicted_matrix2 = pd.DataFrame(Coords.apply(lambda row: R.from_euler(
        'zyx', [row['z'], row['y2'], row['x']]).as_matrix(), axis=1))
    predicted_matrix3 = pd.DataFrame(Coords.apply(lambda row: R.from_euler(
        'zyx', [row['z'], row['y'], row['x']]).as_matrix(), axis=1))

    predicted_matrix1.columns = ['PMatrix']
    predicted_matrix2.columns = ['PMatrix']
    predicted_matrix3.columns = ['PMatrix']
    real_matrix = rot_matrix(labels_path)
    real_matrix.columns = ['RMatrix']
    predicted_matrix1 = pd.concat([real_matrix, predicted_matrix1], axis=1)
    predicted_matrix1.columns = ['RMatrix', 'PMatrix']
    predicted_matrix1['R_0'] = predicted_matrix1.apply(
        lambda row: row['PMatrix'].dot(row['RMatrix'].T), axis=1)
    predicted_matrix1[['R_0Euler_x', 'R_0Euler_y', 'R_0Euler_z']] = predicted_matrix1.apply(
        lambda row: R.from_matrix(row['R_0']).as_euler('xyz', degrees=True), axis=1).tolist()
    predicted_matrix1[['vec_x', 'vec_y', 'vec_z']] = predicted_matrix1.apply(
        lambda row: R.from_matrix(row['R_0']).as_rotvec(), axis=1).tolist()

    predicted_matrix2 = pd.concat([real_matrix, predicted_matrix2], axis=1)
    predicted_matrix2.columns = ['RMatrix', 'PMatrix']
    predicted_matrix2['R_0'] = predicted_matrix2.apply(
        lambda row: row['PMatrix'].dot(row['RMatrix'].T), axis=1)
    predicted_matrix2[['R_0Euler_x', 'R_0Euler_y', 'R_0Euler_z']] = predicted_matrix2.apply(
        lambda row: R.from_matrix(row['R_0']).as_euler('xyz', degrees=True), axis=1).tolist()
    predicted_matrix2[['vec_x', 'vec_y', 'vec_z']] = predicted_matrix2.apply(
        lambda row: R.from_matrix(row['R_0']).as_rotvec(), axis=1).tolist()

    predicted_matrix3 = pd.concat([real_matrix, predicted_matrix3], axis=1)
    predicted_matrix3.columns = ['RMatrix', 'PMatrix']
    predicted_matrix3['R_0'] = predicted_matrix3.apply(
        lambda row: row['PMatrix'].dot(row['RMatrix'].T), axis=1)
    predicted_matrix3[['R_0Euler_x', 'R_0Euler_y', 'R_0Euler_z']] = predicted_matrix3.apply(
        lambda row: R.from_matrix(row['R_0']).as_euler('xyz', degrees=True), axis=1).tolist()
    predicted_matrix3[['vec_x', 'vec_y', 'vec_z']] = predicted_matrix3.apply(
        lambda row: R.from_matrix(row['R_0']).as_rotvec(), axis=1).tolist()

    predicted_matrix1['rotation'] = np.sqrt(
        predicted_matrix1['vec_x']**2+predicted_matrix1['vec_y']**2+predicted_matrix1['vec_z']**2)
    predicted_matrix1['vec_x'] = predicted_matrix1['vec_x'] / \
        predicted_matrix1['rotation']
    predicted_matrix1['vec_y'] = predicted_matrix1['vec_y'] / \
        predicted_matrix1['rotation']
    predicted_matrix1['vec_z'] = predicted_matrix1['vec_z'] / \
        predicted_matrix1['rotation']

    predicted_matrix2['rotation'] = np.sqrt(
        predicted_matrix2['vec_x']**2+predicted_matrix2['vec_y']**2+predicted_matrix2['vec_z']**2)
    predicted_matrix2['vec_x'] = predicted_matrix2['vec_x'] / \
        predicted_matrix2['rotation']
    predicted_matrix2['vec_y'] = predicted_matrix2['vec_y'] / \
        predicted_matrix2['rotation']
    predicted_matrix2['vec_z'] = predicted_matrix2['vec_z'] / \
        predicted_matrix2['rotation']

    predicted_matrix3['rotation'] = np.sqrt(
        predicted_matrix3['vec_x']**2+predicted_matrix3['vec_y']**2+predicted_matrix3['vec_z']**2)
    predicted_matrix3['vec_x'] = predicted_matrix3['vec_x'] / \
        predicted_matrix3['rotation']
    predicted_matrix3['vec_y'] = predicted_matrix3['vec_y'] / \
        predicted_matrix3['rotation']
    predicted_matrix3['vec_z'] = predicted_matrix3['vec_z'] / \
        predicted_matrix3['rotation']
    real_matrix[['R_0Euler_x', 'R_0Euler_y', 'R_0Euler_z']] = real_matrix.apply(
        lambda row: R.from_matrix(row['RMatrix']).as_euler('xyz', degrees=True), axis=1).tolist()

    return Coords, predicted_matrix1, predicted_matrix2, predicted_matrix3, real_matrix


def rot_matrix(labels_path):
    Coords = pd.read_csv(labels_path)
    real_matrix = pd.DataFrame(Coords.apply(lambda row: R.from_quat(
        [row['quat_x'], row['quat_y'], row['quat_z'], row['quat_w']]).as_matrix(), axis=1))
    return real_matrix


def regression(output, path_vae_param, dataset_path, labels_path, epoch_id=None):
    projected_mus, labels = get_cryo(
        output, dataset_path, labels_path, n_pca_components=3)
    mus = latent_projection(
        output, path_vae_param, dataset_path=dataset_path, epoch_id=epoch_id)
    projected_mus = pd.DataFrame(projected_mus)
    projected_mus.columns = ['x1', 'x2', 'x3']
    Coords = pd.DataFrame(projected_mus.apply(lambda row: cs.cart2sp(
        x=row['x1'], y=row['x2'], z=row['x3']), axis=1))
    Coords = pd.DataFrame(Coords)
    Coords[['r', 'theta', 'phi']] = Coords[0].tolist()

    projected_mus = pd.concat([projected_mus, Coords], axis=0)
    return projected_mus, labels, mus


def wigner(alpha):
    return np.array([[np.cos(alpha), 0, np.sin(alpha)], [0, 1, 0], [-np.sin(alpha), 0, np.cos(alpha)]])


def wigner_matrix(rotation):
    J = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    z1, y, z2 = R.from_matrix(rotation).as_euler('zyz', degrees=True)
    return np.dot(np.dot(np.dot(np.dot(wigner(z1), J), wigner(y)), J), wigner(z2))


def norm(vecteur):
    n = len(vecteur)
    norme = 0
    for i in range(n):
        norme += vecteur[i]**2
    return np.sqrt(norme)


def apply_wigner(output, dataset_path, labels_path, liste):
    projected_mus, labels = get_cryo(
        output, dataset_path, labels_path, n_pca_components=3)
    labels['matrix'] = labels.apply(lambda row: R.from_quat(
        [row['quat_x'], row['quat_y'], row['quat_z'], row['quat_w']]).as_matrix(), axis=1)
    L = []
    for couple in liste:
        L.append((np.dot(wigner_matrix(np.dot(labels.iloc[couple[1]]['matrix'],
                                              np.linalg.inv(labels.iloc[couple[0]]['matrix']))), projected_mus[couple[0]])-projected_mus[couple[1]])/norm(projected_mus[couple[1]]))
    return L


def apply_wigner_all_points(output, dataset_path, labels_path, N):
    L = []
    for i in range(N):
        for j in range(i+1, N):
            L.append([i, j])
    projected_mus, labels = get_cryo(
        output, dataset_path, labels_path, n_pca_components=3)
    labels['matrix'] = labels.apply(lambda row: R.from_quat(
        [row['quat_x'], row['quat_y'], row['quat_z'], row['quat_w']]).as_matrix(), axis=1)
    M = []
    for couple in L:
        M.append((np.dot(wigner_matrix(np.dot(labels.iloc[couple[1]]['matrix'],
                                              np.linalg.inv(labels.iloc[couple[0]]['matrix']))), projected_mus[couple[0]])-projected_mus[couple[1]])/norm(projected_mus[couple[1]]))
    return M


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
