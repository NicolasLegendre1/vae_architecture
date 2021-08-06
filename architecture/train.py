# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:36:09 2021

@author: NLEGENDN
"""
import warnings
import torch
import torch.utils.data
import torch.optim
from torch.nn import functional as F
import torch.autograd

from hyperopt import hp
import ray
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.logger import CSVLogger, JsonLogger
from ray.tune import Trainable
from ray import tune

from copy import deepcopy
import time
import random
import numpy as np
import os
import logging


import nn
import cryo_dataset as ds
import train_utils
import losse

import sys
DEBUG = False
arguments = sys.argv


CUDA = torch.cuda.is_available()
if CUDA:
    CRYO_TRAIN_VAL_DIR = os.getcwd() + "/Cryo/VAE_Cryo_V3/Data/"
    path_vae = "Cryo/VAE_Cryo_V3/vae_parameters.json"
    path_data = "Cryo/VAE_Cryo_V3/data_parameters.json"
else:
    CRYO_TRAIN_VAL_DIR = os.getcwd() + "\\Data\\"
    path_vae = "vae_parameters.json"
    path_data = "data_parameters.json"


dataset_name = "4points_3d"
#dataset_name = "4points"

if len(arguments) > 1:
    Dataset = arguments[1]
else:
    Dataset = ''

if len(arguments) > 2:
    img_size = int(arguments[2])


warnings.filterwarnings("ignore")
FOLDER_NAME = dataset_name

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

WITH_RAY = True
CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
OUTPUT_DIR = "/scratch/pr-kdd-1/NicolasLegendre/Cryo"

print(os.getcwd())


PATHS, SHAPES, CONSTANTS, SEARCH_SPACE, _ = ds.load_parameters(
    path_vae)

CONSTANTS["is_3d"] = True
CONSTANTS["img_shape"] = SHAPES[dataset_name]
CONSTANTS['conv_dim'] = 3 if CONSTANTS["is_3d"] else 2

RECONSTRUCTIONS = SEARCH_SPACE['reconstructions']
REGULARIZATIONS = CONSTANTS['regularizations']

# Seed
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

NN_Architecture = ['img_shape', 'latent_dim',
                   'nn_type', 'with_sigmoid', 'n_enc_lay', 'n_dec_lay',
                   "n_gan_lay", "skip_z", "latent_space_definition", 'is_3d']

N_EPOCHS = 300

LR = 15e-6
if 'adversarial' in RECONSTRUCTIONS:
    LR = 0.001  # 0.002 # 0.0002


Train_params = ['dataset_name', 'frac_val', 'lr', 'batch_size',
                'beta1', 'beta2', 'weights_init', 'reconstructions',
                'regularizations', 'lambda_regu', 'lambda_adv', 'class_2d']

SEARCH_PARAMS = SEARCH_SPACE.keys()
CONSTANTS_PARAMS = CONSTANTS.keys()
PRINT_INTERVAL = 2
CKPT_PERIOD = 5


class Config(object):
    def __init__(self, search_space):
        self.search_space = deepcopy(search_space)

    def get(self, arg):
        return self.search_space[arg]

    def get_under_dic(self, list_arg):
        new_dic = {}
        for key in list_arg:
            if key in self.search_space:
                new_dic[key] = self.search_space[key]
        return new_dic


def get_under_dic_cons(const, list_arg):
    new_dic = {}
    for key in list_arg:
        if key in const:
            new_dic[key] = const[key]
    return new_dic


def reinitiate_constante(dics):
    new_dic = {}
    for key in CONSTANTS_PARAMS:
        new_dic[key] = dics[key]
    return new_dic


class Train(Trainable):

    def _setup(self, config):
        reinitiate_constante(CONSTANTS)
        CONSTANTS.update(config)
        train_params = get_under_dic_cons(CONSTANTS, Train_params)
        nn_architecture = get_under_dic_cons(CONSTANTS, NN_Architecture)
        nn_architecture["data_dim"] = CONSTANTS["dim_data"]
        datasets = ds.open_dataset(
            CRYO_TRAIN_VAL_DIR+PATHS[dataset_name],
            CONSTANTS["img_shape"][-1], CONSTANTS["is_3d"])
        _, _, train_loader, val_loader = \
            ds.split_dataset(datasets, CONSTANTS["batch_size"],
                             CONSTANTS["frac_val"])
        m, o, s, t, v = train_utils.init_training(
            self.logdir, train_params, CONSTANTS)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.modules = modules
        self.optimizers = optimizers
        self.start_epoch = start_epoch

        self.train_losses_all_epochs = train_losses_all_epochs
        self.val_losses_all_epochs = val_losses_all_epochs
        self.train_params = train_params
        self.nn_architecture = nn_architecture
        self.latent_space_definition = CONSTANTS["latent_space_definition"]

    def _train_iteration(self):
        """
        One epoch.
        - modules: a dict with the bricks of the model,
        eg. encoder, decoder, discriminator, depending on the architecture
        - optimizers: a dict with optimizers corresponding to each module.
        """

        start = time.time()

        epoch = self._iteration
        nn_architecture = self.nn_architecture
        train_params = self.train_params

        lambda_regu = train_params['lambda_regu']
        lambda_adv = train_params['lambda_adv']
        for module in self.modules.values():
            module.train()

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        if 'adversarial' in train_params['reconstructions']:
            total_loss_discriminator = 0
            total_loss_generator = 0
        total_loss = 0

        n_data = len(self.train_loader.dataset)
        n_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(
                self.train_loader):
            if DEBUG and batch_idx < n_batches - 3:
                continue

            batch_data = batch_data.to(DEVICE)
            n_batch_data = len(batch_data)

            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

            encoder = self.modules['encoder']
            decoder = self.modules['decoder']

            mu, logvar, z_nn, matrix = encoder(batch_data)
            z = z_nn.to(DEVICE)
            batch_recon, scale_b = decoder(z)

            z_from_prior = nn.sample_from_prior(
                nn_architecture['latent_dim'],
                n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, _ = decoder(
                z_from_prior)

            if 'adversarial' in train_params['reconstructions']:
                # From:
                # Autoencoding beyond pixels using a learned similarity metric
                # arXiv:1512.09300v2
                discriminator = self.modules['discriminator_reconstruction']
                real_labels = torch.full(
                    (n_batch_data, 1), 1, device=DEVICE, dtype=torch.float)
                fake_labels = torch.full(
                    (n_batch_data, 1), 0, device=DEVICE, dtype=torch.float)

                # -- Update DiscriminatorGan
                labels_data, h_data, _ = discriminator(
                    batch_data)

                labels_recon, h_recon, h_logvar_recon = discriminator(
                    batch_recon.detach())
                labels_from_prior, _, _ = discriminator(
                    batch_from_prior.detach())

                loss_dis_data = F.binary_cross_entropy(
                    labels_data,
                    real_labels)
                loss_dis_from_prior = F.binary_cross_entropy(
                    labels_from_prior,
                    fake_labels)

                # TODO(nina): add loss_dis_recon
                loss_discriminator = lambda_adv * (
                    loss_dis_data
                    + loss_dis_from_prior)

                # Fill gradients on discriminator only
                loss_discriminator.backward(retain_graph=True)

                # Need to do optimizer step here, as gradients
                # of the reconstruction with discriminator features
                # may fill the discriminator's weights and we do not
                # update the discriminator with the reconstruction loss.
                self.optimizers['discriminator_reconstruction'].step()

                # -- Update Generator/DecoderGAN
                # Note that we need to do a forward pass with detached vars
                # in order not to propagate gradients through the encoder
                batch_recon_detached, _ = decoder(z.detach())
                # Note that we don't need to do it for batch_from_prior
                # as it doesn't come from the encoder

                labels_recon, _, _ = discriminator(
                    batch_recon_detached)
                labels_from_prior, _, _ = discriminator(
                    batch_from_prior)

                loss_generator_recon = F.binary_cross_entropy(
                    labels_recon,
                    real_labels)

                # TODO(nina): add loss_generator_from_prior
                loss_generator = lambda_adv * loss_generator_recon

                # Fill gradients on generator only
                loss_generator.backward()

            if 'mse_on_intensities' in train_params['reconstructions']:
                loss_reconstruction = losse.mse_on_intensities(
                    batch_data, batch_recon, scale_b)

                # Fill gradients on encoder and generator
                loss_reconstruction.backward(retain_graph=True)

            if 'bce_on_intensities' in train_params['reconstructions']:
                loss_reconstruction = losse.bce_on_intensities(
                    batch_data, batch_recon, scale_b)

                # Fill gradients on encoder and generator
                loss_reconstruction.backward(retain_graph=True)

            if 'mse_on_features' in train_params['reconstructions']:
                # TODO(nina): Investigate stat interpretation
                # of using the logvar from the recon
                loss_reconstruction = losse.mse_on_features(
                    h_recon, h_data, h_logvar_recon)
                # Fill gradients on encoder and generator
                # but not on discriminator
                loss_reconstruction.backward(retain_graph=True)

            if 'kullbackleibler' in train_params['regularizations']:
                loss_regularization = lambda_regu * losse.kullback_leibler(
                    mu, logvar)
                # Fill gradients on encoder only
                loss_regularization.backward()

            if 'kullbackleibler_circle' in train_params['regularizations']:
                loss_regularization = losse.kullback_leibler_circle(
                    mu, logvar)
                # Fill gradients on encoder only
                loss_regularization.backward()

            if 'on_circle' in train_params['regularizations']:
                loss_regularization = losse.on_circle(mu, logvar)
                # Fill gradients on encoder only
                loss_regularization.backward()

            self.optimizers['encoder'].step()
            self.optimizers['decoder'].step()

            loss = loss_reconstruction + loss_regularization
            if 'adversarial' in train_params['reconstructions']:
                loss += loss_discriminator + loss_generator

            if batch_idx % PRINT_INTERVAL == 0:
                # TODO(nina): Why didn't we need .mean() on 64x64?
                if 'adversarial' in train_params['reconstructions']:
                    self.print_train_logs(
                        epoch,
                        batch_idx, n_batches, n_data, n_batch_data,
                        loss, loss_reconstruction, loss_regularization,
                        loss_discriminator, loss_generator,
                        labels_data.mean(),
                        labels_recon.mean(),
                        labels_from_prior.mean())
                else:
                    self.print_train_logs(
                        epoch,
                        batch_idx, n_batches, n_data, n_batch_data,
                        loss, loss_reconstruction, loss_regularization)

            # enregistrer les donnees

            total_loss_reconstruction += loss_reconstruction.item()
            total_loss_regularization += loss_regularization.item()
            if 'adversarial' in train_params['reconstructions']:
                total_loss_discriminator += loss_discriminator.item()
                total_loss_generator += loss_generator.item()
            total_loss += loss.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        if 'adversarial' in train_params['reconstructions']:
            average_loss_discriminator = total_loss_discriminator / n_data
            average_loss_generator = total_loss_generator / n_data
        average_loss = total_loss / n_data

        logging.info('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, average_loss))

        end = time.time()

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        if 'adversarial' in train_params['reconstructions']:
            train_losses['discriminator'] = average_loss_discriminator
            train_losses['generator'] = average_loss_generator
        train_losses['total'] = average_loss
        train_losses['total_time'] = end - start
        # mlflow.log_metrics(train_losses)

        self.train_losses_all_epochs.append(train_losses)

    def _train(self):
        self._train_iteration()
        return self._test()

    def _test(self):

        start = time.time()

        epoch = self._iteration
        nn_architecture = self.nn_architecture
        train_params = self.train_params

        for module in self.modules.values():
            module.eval()

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        if 'adversarial' in train_params['reconstructions']:
            total_loss_discriminator = 0
            total_loss_generator = 0
        total_loss = 0

        n_data = len(self.val_loader.dataset)
        n_batches = len(self.val_loader)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                if DEBUG and batch_idx < n_batches - 3:
                    continue
                batch_data = batch_data.to(DEVICE)
                n_batch_data = batch_data.shape[0]

                encoder = self.modules['encoder']
                decoder = self.modules['decoder']

                mu, logvar, z_nn, matrix = encoder(batch_data)
                z = z_nn.to(DEVICE)
                batch_recon, scale_b = decoder(z)

                z_from_prior = nn.sample_from_prior(
                    nn_architecture['latent_dim'],
                    n_samples=n_batch_data).to(DEVICE)
                batch_from_prior, _ = decoder(
                    z_from_prior)

                if 'adversarial' in train_params['reconstructions']:
                    discriminator = self.modules[
                        'discriminator_reconstruction']
                    real_labels = torch.full(
                        (n_batch_data, 1), 1, device=DEVICE)
                    fake_labels = torch.full(
                        (n_batch_data, 1), 0, device=DEVICE)

                    # -- Compute DiscriminatorGan Loss
                    labels_data, h_data, _ = discriminator(batch_data)
                    labels_recon, h_recon, h_logvar_recon = discriminator(
                        batch_recon.detach())
                    labels_from_prior, _, _ = discriminator(
                        batch_from_prior.detach())
                    loss_dis_data = F.binary_cross_entropy(
                        labels_data,
                        real_labels.float())
                    loss_dis_from_prior = F.binary_cross_entropy(
                        labels_from_prior,
                        fake_labels.float())
                    loss_discriminator = (
                        loss_dis_data
                        + loss_dis_from_prior)

                    batch_recon_detached, _ = decoder(z.detach())

                    labels_recon, _, _ = discriminator(
                        batch_recon_detached)
                    labels_from_prior, _, _ = discriminator(
                        batch_from_prior)

                    loss_generator_recon = F.binary_cross_entropy(
                        labels_recon,
                        real_labels.float())

                    loss_generator = loss_generator_recon

                if 'mse_on_intensities' in train_params['reconstructions']:
                    loss_reconstruction = losse.mse_on_intensities(
                        batch_data, batch_recon, scale_b)

                if 'bce_on_intensities' in train_params['reconstructions']:
                    loss_reconstruction = losse.bce_on_intensities(
                        batch_data, batch_recon, scale_b)

                if 'mse_on_features' in train_params['reconstructions']:

                    loss_reconstruction = losse.mse_on_features(
                        h_recon, h_data, h_logvar_recon)

                if 'kullbackleibler' in train_params['regularizations']:
                    loss_regularization = losse.kullback_leibler(
                        mu, logvar)
                if 'kullbackleibler_circle' in train_params['regularizations']:
                    loss_regularization = losse.kullback_leibler_circle(
                        mu, logvar)

                if 'on_circle' in train_params['regularizations']:
                    loss_regularization = losse.on_circle(
                        mu, logvar)

                loss = loss_reconstruction + loss_regularization
                if 'adversarial' in train_params['reconstructions']:
                    loss += loss_discriminator + loss_generator

                total_loss_reconstruction += loss_reconstruction.item()
                total_loss_regularization += loss_regularization.item()
                if 'adversarial' in train_params['reconstructions']:
                    total_loss_discriminator += loss_discriminator.item()
                    total_loss_generator += loss_generator.item()
                total_loss += loss.item()

                if batch_idx == n_batches - 1:
                    batch_data = batch_data.cpu().numpy()
                    batch_recon = batch_recon.cpu().numpy()
                    batch_from_prior = batch_from_prior.cpu().numpy()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        if 'adversarial' in train_params['reconstructions']:
            average_loss_discriminator = total_loss_discriminator / n_data
            average_loss_generator = total_loss_generator / n_data
        average_loss = total_loss / n_data
        print('====> Val set loss: {:.4f}'.format(average_loss))

        end = time.time()

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        if 'adversarial' in train_params['reconstructions']:
            val_losses['discriminator'] = average_loss_discriminator
            val_losses['generator'] = average_loss_generator
        val_losses['total'] = average_loss
        val_losses['total_time'] = end - start

        if np.isnan(average_loss) or np.isinf(average_loss):
            raise ValueError('Val loss is too large.')
        self.val_losses_all_epochs.append(val_losses)
        return {'average_loss': average_loss}

    def _save(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = self.logdir
        epoch = self._iteration
        train_utils.save_checkpoint(
            dir_path=checkpoint_dir,
            nn_architecture=self.nn_architecture,
            train_params=self.train_params,
            epoch=epoch,
            modules=self.modules,
            optimizers=self.optimizers,
            train_losses_all_epochs=self.train_losses_all_epochs,
            val_losses_all_epochs=self.val_losses_all_epochs)
        checkpoint_path = os.path.join(
            checkpoint_dir, 'epoch_%d_checkpoint.pth' % epoch)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        epoch_id = None  # HACK: restore last one
        train_dir = os.path.dirname(checkpoint_path)
        output = os.path.dirname(train_dir)
        for module_name, module in self.modules.items():
            self.modules[module_name] = train_utils.load_module_state(
                output=output,
                module_name=module_name,
                module=module,
                epoch_id=epoch_id)

    @staticmethod
    def print_train_logs(self,
                         epoch,
                         batch_idx, n_batches, n_data, n_batch_data,
                         loss,
                         loss_reconstruction, loss_regularization,
                         loss_discriminator=0, loss_generator=0,
                         dx=0, dgex=0, dgz=0):

        loss = loss / n_batch_data
        loss_reconstruction = loss_reconstruction / n_batch_data
        loss_regularization = loss_regularization / n_batch_data
        loss_discriminator = loss_discriminator / n_batch_data
        loss_generator = loss_generator / n_batch_data

        string_base = ('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'
                       + '\nReconstruction: {:.6f}, Regularization: {:.6f}')

        if 'adversarial' in CONSTANTS['reconstructions']:
            string_base += (
                ', Discriminator: {:.6f}; Generator: {:.6f},'
                + 'D(x): {:.3f}, D(G(E(x))): {:.3f}, D(G(z)): {:.3f}')

        if 'adversarial' not in CONSTANTS['reconstructions']:
            logging.info(
                string_base.format(
                    epoch, batch_idx * n_batch_data, n_data,
                    100. * batch_idx / n_batches,
                    loss, loss_reconstruction, loss_regularization))
        else:
            logging.info(
                string_base.format(
                    epoch, batch_idx * n_batch_data, n_data,
                    100. * batch_idx / n_batches,
                    loss, loss_reconstruction, loss_regularization,
                    loss_discriminator, loss_generator,
                    dx, dgex, dgz))


def init():
    logging.getLogger().setLevel(logging.INFO)
    logging.info('start')


if __name__ == "__main__":
    init()
    if WITH_RAY:
        ray.init()
        search_space = {
            "dataset_name": dataset_name,
            "lr": hp.loguniform(
                "lr",
                low=np.log(0.0001),
                high=np.log(0.1)),
            "latent_dim": hp.choice("latent_dim", [6]),
            "n_enc_lay": hp.choice("n_enc_lay", [4, 5, 6]),
            "n_dec_lay": hp.choice("n_dec_lay", [4, 5, 6]),
            "lambda_regu": hp.loguniform(
                "lambda_regu",
                low=np.log(0.001), high=np.log(0.5)),
            "lambda_adv": hp.loguniform(
                "lambda_adv",
                low=np.log(0.001), high=np.log(0.5)),
            "batch_size": 20,
            "n_gan_lay": 4,
            "regu_factor": hp.loguniform(
                'regu_factor',
                low=np.log(0.001), high=np.log(0.5)),
            "skip_z": hp.choice("skip_z", [False, True]),
            "latent_space_definition": 1
        }

        hyperband_sched = AsyncHyperBandScheduler(
            grace_period=5,
            time_attr='training_iteration',
            metric='average_loss',
            brackets=1,
            reduction_factor=4,
            max_t=N_EPOCHS,
            mode='min')

        hyperopt_search = HyperOptSearch(
            search_space,
            metric='average_loss',
            mode='min',
            max_concurrent=257)

        analysis = tune.run(
            Train,
            local_dir=OUTPUT_DIR,
            name=FOLDER_NAME,
            scheduler=hyperband_sched,
            search_alg=hyperopt_search,
            loggers=[JsonLogger, CSVLogger],
            queue_trials=True,
            reuse_actors=False,
            **{
                'stop': {
                    'training_iteration': N_EPOCHS,
                },
                'resources_per_trial': {
                    'cpu': 4,
                    'gpu': 1
                },
                'max_failures': 1,
                'num_samples': 300,
                'checkpoint_freq': CKPT_PERIOD,
                'checkpoint_at_end': True,
                'config': search_space})

    else:
        model = Train()
        model._train_iteration()
        model._test()
