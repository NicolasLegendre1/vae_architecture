import logging
import numpy as np
import os

from hyperopt import hp
import ray
from ray.tune import Trainable
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.logger import CSVLogger, JsonLogger
from ray import tune

import time
import torch

import initialization_pipeline as inipip
import losses
import neural_network
import train_utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')
OUTPUT_DIR = "/scratch/pr-kdd-1/NicolasLegendre/Cryo"


CRYO_TRAIN_VAL_DIR = inipip.initialization_path(CUDA)

META_CONFIG = inipip.load_meta_config(CRYO_TRAIN_VAL_DIR+"meta_config.json")

FOLDER_NAME = META_CONFIG["dataset_name"] + str(META_CONFIG["size"]) + \
    str(META_CONFIG["dimension"]) + \
    META_CONFIG["latent_space"] + META_CONFIG["mean_mode"]

CONFIG = inipip.choose_dimension(
    META_CONFIG, CRYO_TRAIN_VAL_DIR)


def initialization_img_shape(meta_config):
    return (1,)+(meta_config["size"],)*meta_config["dimension"]


CONFIG["img_shape"] = initialization_img_shape(META_CONFIG)

CONFIG[META_CONFIG["dataset"]
       ] = CONFIG[META_CONFIG["dataset"]].replace("\\", "/")


class Train(Trainable):

    def _setup(self, search_space):
        config = inipip.join_dics([CONFIG, search_space])
        dataset = inipip.initialization_dataset(
            META_CONFIG, CRYO_TRAIN_VAL_DIR + CONFIG[META_CONFIG["dataset"]])
        _, _, train_loader, val_loader = \
            inipip.split_dataset(dataset, config)
        m, o, s, t, v = train_utils.init_training(
            self.logdir, config)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.modules = modules
        self.optimizers = optimizers
        self.start_epoch = start_epoch
        self.train_losses_all_epochs = train_losses_all_epochs
        self.val_losses_all_epochs = val_losses_all_epochs
        self.config = config

    def _train_iteration(self):
        """
        One epoch.
        - modules: a dict with the bricks of the model,
        eg. encoder, decoder, discriminator, depending on the architecture
        - optimizers: a dict with optimizers corresponding to each module.
        """

        start = time.time()

        epoch = self._iteration

        lambda_regu = self.config['lambda_regu']
        lambda_adv = self.config['lambda_adv']
        for module in self.modules.values():
            module.train()

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_loss = 0

        n_data = len(self.train_loader.dataset)
        n_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(
                self.train_loader):
            if META_CONFIG["debug"] and batch_idx < n_batches - 3:
                continue

            batch_data = batch_data.to(DEVICE)
            n_batch_data = len(batch_data)

            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

            encoder = self.modules['encoder']
            decoder = self.modules['decoder']

            self.latent_dim = decoder.latent_dim

            z_nn, matrix = encoder(batch_data)
            z = z_nn.to(DEVICE)
            batch_recon, scale_b = decoder(z)
            # z_from_prior = neural_network.sample_from_prior(
            # self.latent_dim, n_samples=n_batch_data).to(DEVICE)
            #batch_from_prior, _ = decoder(z_from_prior)

            if 'mse_on_intensities' in self.config['reconstructions']:
                loss_reconstruction = losses.mse_on_intensities(
                    batch_data, batch_recon, scale_b)

                # Fill gradients on encoder and generator
                loss_reconstruction.backward(retain_graph=True)

            if "bce_on_intensities" in self.config['reconstructions']:

                loss_reconstruction = losses.bce_on_intensities(
                    batch_data, batch_recon, scale_b)

                # Fill gradients on encoder and generator
                loss_reconstruction.backward(retain_graph=True)

            if "kullbackleibler" in self.config['regularizations']:
                loss_regularization = encoder.kl()[0].sum()
                # Fill gradients on encoder only
                loss_regularization.backward(retain_graph=True)

            self.optimizers['encoder'].step()
            self.optimizers['decoder'].step()

            loss = loss_reconstruction + loss_regularization

            if batch_idx % META_CONFIG["print_interval"] == 0:
                self.print_train_logs(self, epoch, batch_idx,
                                      n_batches, n_data,
                                      n_batch_data, loss,
                                      loss_reconstruction,
                                      loss_regularization)

            # enregistrer les donnees

            total_loss_reconstruction += loss_reconstruction.item()
            total_loss_regularization += loss_regularization.item()
            total_loss += loss.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_loss = total_loss / n_data

        logging.info('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, average_loss))

        end = time.time()

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
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

        for module in self.modules.values():
            module.eval()

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_loss = 0

        n_data = len(self.val_loader.dataset)
        n_batches = len(self.val_loader)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                if META_CONFIG["debug"] and batch_idx < n_batches - 3:
                    continue
                batch_data = batch_data.to(DEVICE)
                n_batch_data = batch_data.shape[0]

                encoder = self.modules['encoder']
                decoder = self.modules['decoder']

                z_nn, matrix = encoder(batch_data)
                z = z_nn.to(DEVICE)
                batch_recon, scale_b = decoder(z)

                # z_from_prior = neural_network.sample_from_prior(
                # self.latent_dim, n_samples=n_batch_data).to(DEVICE)
                #batch_from_prior, _ = decoder(z_from_prior)

                if 'mse_on_intensities' in self.config['reconstructions']:
                    loss_reconstruction = losses.mse_on_intensities(
                        batch_data, batch_recon, scale_b)

                if 'bce_on_intensities' in self.config['reconstructions']:
                    loss_reconstruction = losses.bce_on_intensities(
                        batch_data, batch_recon, scale_b)

                if 'kullbackleibler' in self.config['regularizations']:
                    loss_regularization = encoder.kl()[0].sum()

                loss = loss_reconstruction + loss_regularization

                total_loss_reconstruction += loss_reconstruction.item()
                total_loss_regularization += loss_regularization.item()
                total_loss += loss.item()

                if batch_idx == n_batches - 1:
                    batch_data = batch_data.cpu().numpy()
                    batch_recon = batch_recon.cpu().numpy()
                    #batch_from_prior = batch_from_prior.cpu().numpy()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_loss = total_loss / n_data
        print('====> Val set loss: {:.4f}'.format(average_loss))

        end = time.time()

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
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
            config=self.config,
            meta_config=META_CONFIG,
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

        if 'adversarial' not in self.config['reconstructions']:
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
    if META_CONFIG["with_ray"]:
        ray.init()
        search_space = {
            "dataset_name": META_CONFIG["dataset_name"],
            "lr": hp.loguniform(
                "lr",
                low=np.log(0.0001),
                high=np.log(0.1)),
            "n_enc_lay": hp.choice("n_enc_lay", [2, 3, 4]),
            "n_dec_lay": hp.choice("n_dec_lay", [2, 3, 4]),
            "latent_dim": hp.choice("latent_dim", [1, 2, 3]),
            "lambda_regu": 1,
            "lambda_adv": hp.loguniform(
                "lambda_adv",
                low=np.log(0.0001), high=np.log(0.001)),
            "batch_size": 8,
            "regu_factor": 1,
            "skip_z": hp.choice("skip_z", [False, True]),
            # "mean_mode": "s2s2",
            "wigner_dim": hp.choice("wigner_dim", [1, 2, 3])
        }

        hyperband_sched = AsyncHyperBandScheduler(
            grace_period=5,
            time_attr='training_iteration',
            metric='average_loss',
            brackets=1,
            reduction_factor=4,
            max_t=META_CONFIG["n_epochs"],
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
                    'training_iteration': META_CONFIG["n_epochs"],
                },
                'resources_per_trial': {
                    'cpu': 4,
                    'gpu': 1
                },
                'max_failures': 1,
                'num_samples': 200,
                'checkpoint_freq': META_CONFIG["ckpt_period"],
                'checkpoint_at_end': True,
                'config': search_space})

    else:
        search_space = {
            "dataset_name": META_CONFIG["dataset_name"],
            "lr": 0.1,
            "wigner_dim": 2,
            "conv_dim": 2,
            "n_enc_lay": 5,
            "n_dec_lay":  2,
            "lambda_regu": 1,
            "lambda_adv": 0.01,
            "batch_size": 8,
            "n_gan_lay": 1,
            "regu_factor": 1,
            "latent_dim": 3,
            "skip_z": True
        }
        model = Train(search_space)
        batch_data = model._train_iteration()
        batch_data = model._train_iteration()
        batch_data = model._train_iteration()
        model._test()
        model._test()
        model._test()
