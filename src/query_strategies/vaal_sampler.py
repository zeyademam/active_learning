import comet_ml
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim  # noqa
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DataLoader, Subset

from .strategy import Strategy
from .vaal_discriminator import Discriminator
from .vae import VAE


class VAALSampler(Strategy):
    """
    Based on https://arxiv.org/pdf/1904.00370.pdf with some tweaks.
    """
    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs):
        super().__init__(train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set,
                         **kwargs)
        # LAZY IMPLEMENTATION. Imagenet uses scale 2 and cifar10 uses scale 1
        if self.num_classes == 10:
            self.latent_scale = 1.0
        elif self.num_classes == 1000:
            self.latent_scale = 2.0
        else:
            raise ValueError("Unsupported dataset")
        self.vae = VAE(z_dim=kwargs["vae_latent_dim"], nc=3, latent_scale=self.latent_scale)
        self.discriminator = Discriminator(z_dim=kwargs["vae_latent_dim"])
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.adversary_param = kwargs[
            "vaal_adversary_param"]  # lambda2 in the VAAL paper: 10 for imagenet
        self.lr_discriminator = kwargs["lr_discriminator"]
        self.lr_vae = kwargs["lr_vae"]

    def query(self, budget):

        idxs_for_query = self.available_query_idxs()
        al_set = Subset(self.al_set, idxs_for_query)
        al_set_loader = DataLoader(al_set, shuffle=False, **self.train_args["loader_te_args"],
                                   drop_last=False)
        self.logger.debug(f'# batches {len(al_set_loader)}')
        self.net.eval()
        self.vae.eval()
        self.discriminator.eval()

        all_preds = []
        all_indices = []

        for batch_idx, (x, y, idxs) in enumerate(al_set_loader):
            with torch.no_grad():
                x = x.to(self.device)
                _, _, _, mu, _ = self.vae(x)
                preds = self.discriminator(mu)
                preds = preds.cpu().data
                all_preds.extend(preds)
                all_indices.extend(idxs.clone())
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1
        # select the points which the discriminator thinks are the most likely to be unlabeled
        query_count = int(min(len(idxs_for_query), budget))
        _, query_idxs = torch.topk(all_preds, query_count)
        labeled_idxs = np.asarray(all_indices)[query_idxs]

        return labeled_idxs, query_count

    def init_network_weights(self):
        super().init_network_weights()
        self.vae.weight_init()
        self.discriminator.weight_init()

    def parallel_train_fn(self, rank):
        weight_paths = self.generate_weight_paths()

        # Setup process groups and train_sampler
        train_set_wrapper = Subset(self.train_set,
                                   self.already_labeled_idxs(boolean=False, shuffle=False))
        unlabeled_set_wrapper = Subset(self.train_set,
                                       self.available_query_idxs(boolean=False, shuffle=False))

        if self.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set_wrapper,
                                                                            num_replicas=self.world_size,
                                                                            rank=rank, shuffle=True)
            unlabeled_data_sampler = torch.utils.data.distributed.DistributedSampler(
                unlabeled_set_wrapper, num_replicas=self.world_size, rank=rank, shuffle=True)

            dist.init_process_group("nccl", rank=rank, world_size=self.world_size)
            torch.cuda.set_device(rank)
            if rank == 0:
                self.comet_experiment = comet_ml.ExistingExperiment(
                    previous_experiment=self.comet_exp_key)

        else:
            train_sampler = None
            unlabeled_data_sampler = None

        # Setup training data loader using train_sampler
        loader_tr_args = self.train_args['loader_tr_args'].copy()
        loader_tr_args["batch_size"] = int(loader_tr_args["batch_size"] / self.world_size)
        loader_tr = DataLoader(train_set_wrapper, shuffle=(train_sampler is None), **loader_tr_args,
                               drop_last=False, sampler=train_sampler, pin_memory=True)
        loader_unlabeled_data = DataLoader(unlabeled_set_wrapper,
                                           shuffle=(unlabeled_data_sampler is None),
                                           **loader_tr_args, drop_last=False,
                                           sampler=unlabeled_data_sampler, pin_memory=True)

        # Setup DDP network for parallel training
        self.net = self.net.to(rank)
        self.vae = self.vae.to(rank)
        self.discriminator = self.discriminator.to(rank)
        self.net.train()
        self.vae.train()
        self.discriminator.train()

        if self.world_size > 1:
            # self.net = DDP(self.net, device_ids=[rank], find_unused_parameters=True)
            self.net = DDP(self.net, device_ids=[rank])

        # Reset training params
        self.es_params["count"] = 0
        self.es_params["success"] = False
        self.es_params["best_perf"] = 0
        step = 0

        # Create training routine components
        optimizer = eval(f"optim.{self.train_args['optimizer']}")(self.net.parameters(),
                                                                  **self.train_args[
                                                                      "optimizer_args"])
        scheduler = eval(f"optim.lr_scheduler.{self.train_args['lr_scheduler']}")(optimizer,
                                                                                  **self.train_args[
                                                                                      "lr_scheduler_args"])

        optim_vae = optim.Adam(self.vae.parameters(), lr=self.lr_vae)
        optim_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.lr_discriminator)
        scheduler_vae = eval(f"optim.lr_scheduler.{self.train_args['lr_scheduler']}")(optim_vae, **
        self.train_args["lr_scheduler_args"])
        scheduler_discriminator = eval(f"optim.lr_scheduler.{self.train_args['lr_scheduler']}")(
            optim_discriminator, **self.train_args["lr_scheduler_args"])

        print(f'Rank {rank} training starts.')
        if self.imbalanced_training:
            criterion = nn.CrossEntropyLoss(weight=self.imb_weights, reduction="mean").to(
                self.device)
        else:
            criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)

        self.logger.info(f"Starting training on round {self.round}")
        for epoch in range(1, self.n_epoch + 1):
            # Freeze batchnorm params if we're doing linear classification or end-to-end training
            # with a
            # pretrained backbone (e.g. transfer learning)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self.net.train()
            self.vae.train()
            self.discriminator.train()
            if self.freeze_feature or ('init_pretrained_ckpt_path' in self.train_args):
                self.net.eval()
            step = self.vaal_train(epoch=epoch, rank=rank, loader_tr=loader_tr, optimizer=optimizer,
                                   criterion=criterion, step=step,
                                   loader_unlabeled_data=loader_unlabeled_data, optim_vae=optim_vae,
                                   optim_discriminator=optim_discriminator)

            scheduler.step()
            scheduler_vae.step()
            scheduler_discriminator.step()
            self.net = self.net.to(rank)

            # Run validation if early stop is required
            if self.validation_and_early_stopping(rank, epoch, weight_paths): break

        msg = f'Sanity Check: Best ckpt of worker rank {rank} occurs on epoch {self.best_epoch}'
        print(msg)
        self.logger.info(msg)
        self.logger.info(f"Finished training on round {self.round}")

        pass

    def vaal_train(self, rank, epoch, loader_tr, optimizer, criterion, step, loader_unlabeled_data,
                   optim_vae, optim_discriminator):

        """Train self.net for a single training epoch.

        Args:
            epoch (int): The current epoch number.
            train_set: The dataset to train on.
            optimizer: The optimizer.
            criterion: The loss criterion.
            step (int): Global training step (i.e. carried over since epoch 0).

        Returns:

        """
        total_loss = 0
        loader_unlabeled_data_iterator = iter(loader_unlabeled_data)

        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            try:
                x_u, _, idxs_u = next(loader_unlabeled_data_iterator)
            except StopIteration:
                loader_unlabeled_data_iterator = iter(loader_unlabeled_data)
                x_u, _, idxs_u = next(loader_unlabeled_data_iterator)

            x_u = x_u.to(self.device, non_blocking=True)

            self.vae.set_crop_seed(np.random.randint(0, 10000))
            # NETWORK STEP
            optimizer.zero_grad()
            out = self.net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            cur_loss = loss.cpu().detach()
            total_loss += cur_loss
            msg = f"\tRound {self.round}, Epoch {epoch},  batch {batch_idx}/{len(loader_tr)}," \
                  f"loss is {cur_loss} on worker rank {rank}"
            if batch_idx % 25 == 0:
                self.logger.info(msg)
                if self.world_size == 1 or rank == 1:
                    print(msg)

            # VAE STEP
            x_crop, recon, z, mu, logvar = self.vae(x)

            unsup_loss = self.vae_loss(x_crop, recon, mu, logvar, 1)

            x_u_crop, unlab_recon, unlab_z, unlab_mu, unlab_logvar = self.vae(x_u)
            transductive_loss = self.vae_loss(x_u_crop, unlab_recon, unlab_mu, unlab_logvar, 1)

            labeled_preds = self.discriminator(mu)
            unlabeled_preds = self.discriminator(unlab_mu)
            lab_real_preds = torch.ones(x.size(0)).to(self.device, non_blocking=True)
            unlab_real_preds = torch.ones(x_u.size(0)).to(self.device, non_blocking=True)

            dsc_loss = self.bce_loss(labeled_preds.squeeze(), lab_real_preds) + self.bce_loss(
                unlabeled_preds.squeeze(), unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + self.adversary_param * dsc_loss
            optim_vae.zero_grad()
            total_vae_loss.backward()
            optim_vae.step()

            # DISCRIMINATOR STEP
            with torch.no_grad():
                _, _, _, mu, _ = self.vae(x)
                _, _, _, unlab_mu, _ = self.vae(x_u)
            labeled_preds = self.discriminator(mu)
            unlabeled_preds = self.discriminator(unlab_mu)

            lab_real_preds = torch.ones(x.size(0))
            unlab_fake_preds = torch.zeros(x_u.size(0))

            lab_real_preds = lab_real_preds.to(self.device, non_blocking=True)
            unlab_fake_preds = unlab_fake_preds.to(self.device, non_blocking=True)

            dsc_loss = self.bce_loss(labeled_preds.squeeze(), lab_real_preds) + self.bce_loss(
                unlabeled_preds.squeeze(), unlab_fake_preds)

            optim_discriminator.zero_grad()
            dsc_loss.backward()
            optim_discriminator.step()

            step += 1

        total_loss /= len(loader_tr)

        return step

    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
