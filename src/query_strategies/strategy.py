import logging
import os

import comet_ml
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim  # noqa
from models.utils import init_params
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from utils.evaluation import evaluate, gather_parallel_eval
from utils.load_pretrained_weights import load_pretrained_weights
from utils.parallel_training_utils import get_free_tcp_port

dirname = os.path.dirname


class Strategy:
    """
    The base class for an active learning query strategy.
    Attributes:
            al_set: Same as train_set but without the the data augmentation.
                Used by the AL strategy to decide which patches of the data to
                query.

            comet_experiment: The Comet Experiment used for monitoring the
                experiment.

            cumulative_cost (float): The exact budget exhausted so far by the
                AL algo. i.e. the number of labeled images.

            eval_idxs (List[int]): The idxs of images from the train set that are used for early
                stopping validation.
                This data is ignored when AL algos query for new patches. We are assumed to have the
                labels for those images a priori.

            init_weights_path (str): The network's weights at every AL round, are either randomly
                initialized or loaded from ckpt if the network was pretrained using SSL or transfer
                learning or other.
                This variable is a path (str) where these initial network weights are saved. If
                None, then the network weights are randomly intialized at every training round.

            idxs_lb (np.ndarray): A one dimensional boolean np array of
                length n_pool containing True at idxs of images that have been
                fully labeled.

            idxs_lb_recent (np.ndarray): A one dimensional boolean np array of
                length n_pool containing True at idxs of images that have been
                most recently labeled on the last call to query.

            n_pool (int): The number of images in the al_set. This also counts
                images considered only for validation (see eval_idxs).

            net: The main classifier model.

            round (int): The current active learning round. Initially the AL
                algorithm starts at round 0 with a randomly sampled inital labeled pool, then
                increases everytime a new batch of images is queried.

            train_args (dict): A Dict containing training parameters and
                pretrained checkpoint paths.

            train_set: An instance of a torch dataset. The train set usually
                includes data augmentation transforms such as random crops,
                flips etc. It is assumed to contain the entire training dataset.
                This dataset is used to train the segmentation network.
    """

    logger = logging.getLogger("ActiveLearning")

    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs, ):
        # Logging params
        self.train_args = train_args
        self.comet_experiment = comet_experiment
        tmp = os.path.basename(os.path.normpath(comet_experiment.url))[:9]
        self.comet_experiment_hash = "debug" if tmp == "." else tmp

        # Dataset params
        self.train_set = train_set
        self.al_set = al_set
        self.test_set = test_set

        self.num_classes = self.al_set.num_classes
        self.logger.info(f"Number of classes: {self.num_classes}")

        # Active Learning General Params
        self.round = 0
        self.cumulative_cost = 0

        # Active Learning Querying params
        self.n_pool = len(self.al_set)
        self.eval_idxs = eval_idxs
        self.idxs_lb = np.zeros(self.n_pool, dtype=np.bool)
        self.idxs_lb_recent = np.zeros(self.n_pool, dtype=np.bool)

        # Active Learning Training and Early Stopping Params.
        self.es_params = {"use_es": False if kwargs["early_stop_patience"] == 0 else True,
                          "patience": kwargs["early_stop_patience"], "count": 0, "success": False,
                          "best_perf": 0, }
        self.n_epoch = kwargs["n_epoch"]
        self.imbalanced_training = train_args.get("imbalanced_training", False)

        # Parallel training args
        self.world_size = kwargs['world_size']

        # Network Params
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.net = net
        self.net_name = kwargs["model"]
        self.query_net = None
        self.feature_net = None
        self.freeze_feature = kwargs["freeze_feature"]

        # Misc
        self.base_ckpt_path = kwargs["ckpt_path"]
        self.exp_name = kwargs["exp_name"]
        self.exp_hash = os.path.basename(os.path.normpath(self.comet_experiment.url))[:9]
        if self.exp_hash == '.': self.exp_hash = "no_comet"

    def available_query_idxs(self, boolean=False, shuffle=True):
        """
            Get the idxs of images that haven't already been fully labeled.

        Args:
            self:

        Returns:

        """
        if boolean:
            idxs_for_query = ~self.idxs_lb
            idxs_for_query[self.eval_idxs] = False
            return idxs_for_query
        else:
            idxs_for_query = np.where(self.idxs_lb == False)[0]
            if shuffle:
                idxs_for_query = np.random.permutation(idxs_for_query)
            idxs_for_query = np.array([x for x in idxs_for_query if x not in self.eval_idxs])
            return idxs_for_query

    def already_labeled_idxs(self, boolean=False, shuffle=False):
        """
            Get the idxs of images that have already beeng labeled

        Args:
            self:

        Returns:

        """
        if boolean:
            return np.copy(self.idxs_lb)
        else:
            idxs_already_labeled = np.argwhere(self.idxs_lb).squeeze()
            if shuffle:
                idxs_already_labeled = np.random.permutation(idxs_already_labeled)
            return idxs_already_labeled

    def generate_weight_paths(self):
        ckpt_dir = os.path.join(self.base_ckpt_path, f"{self.exp_name}_{self.exp_hash}")
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        paths = {"best_ckpt": os.path.join(ckpt_dir, f"best_rd_{self.round}.pth"),
                 "previous_ckpt": os.path.join(ckpt_dir, f"rd_{self.round - 1}.pth"),
                 "current_ckpt": os.path.join(ckpt_dir, f"rd_{self.round}.pth"), }

        return paths

    def init_network_weights(self):
        """This function initializes network weights either randomly or from a pretrained ckpt.
        Returns: None.

        """
        init_ckpt_path = self.train_args.get('init_pretrained_ckpt_path')

        # Reset network weights randomly to make sure that the linear classification head is reset,
        # even if we're using a pretrained backbone.
        self.net.apply(init_params)
        if init_ckpt_path is None:
            self.logger.info("Initialized Network Weights Randomly.")
        else:
            self.logger.info(f"Initializing Network Weights from {init_ckpt_path}")
            skip_key = self.train_args.get('skip_key')
            required_key = self.train_args.get('required_key')
            replace_key = self.train_args.get('replace_key')
            self.logger.info(
                f'required_key = {required_key}, replace_key = {replace_key}, skip_key= '
                f'{skip_key}')
            self.net = load_pretrained_weights(self.net, init_ckpt_path, replace_key=replace_key,
                                               skip_key=skip_key, required_key=required_key)

        self.feature_net = self.net

        pass

    def load_best_ckpt(self):
        best_ckpt_path = self.generate_weight_paths()["best_ckpt"]
        self.logger.info(f"Loading best ckpt so far from: {best_ckpt_path}")
        self.net = load_pretrained_weights(self.net, best_ckpt_path)
        pass

    def query(self, budget):
        pass

    def test(self):
        if not self.test_set:
            self.logger.info(f"Skipped testing loop, no testing dataset found.")
            return
        self.net.to(self.device)
        loader_te_args = self.train_args["loader_te_args"].copy()
        loader_te_args["batch_size"] = int(loader_te_args["batch_size"] / self.world_size)
        test_loader = DataLoader(self.test_set, shuffle=False, **loader_te_args, drop_last=False,
                                 pin_memory=True)
        test_perf_dict = evaluate(test_loader, net=self.net, metric="accuracy",
                                  num_classes=self.num_classes, net_name=self.net_name, )
        test_perf = test_perf_dict['accuracy'].cpu()
        test_top5 = test_perf_dict['top_5_accuracy'].cpu()
        test_byclass = test_perf_dict['accuracy_byclass'].cpu()
        idxs = sorted(range(len(test_byclass)), key=lambda k: test_byclass[k])
        tmp = int(min(5, len(test_byclass)))
        best_classes = {idx: f"{test_byclass[idx].item() * 100:.2f}" for idx in idxs[-tmp:]}
        worst_classes = {idx: f"{test_byclass[idx].item() * 100:.2f}" for idx in idxs[:tmp]}

        self.logger.info(f"Test performance at round {self.round} is {test_perf * 100:.2f}%")
        self.logger.info(
            f"Test performance by class at round {self.round} for the best {tmp} class: "
            f"{best_classes}")
        self.logger.info(
            f"Test performance by class at round {self.round} for the worst {tmp} class: "
            f"{worst_classes}")
        self.logger.info(f"Test top 5 acc at round {self.round} " f"is {test_top5 * 100:.2f}%")

        self.comet_experiment.log_metrics(
            {f"rd_test_accuracy": test_perf, f"rd_test_top5_accuracy": test_top5}, step=self.round)
        self.comet_experiment.log_metrics(
            {f"budget_test_accuracy": test_perf, f'budget_test_top5_accuracy': test_top5, },
            step=self.cumulative_cost, )
        self.comet_experiment.log_asset_data(",".join(f"{e.item():.2f}" for e in test_byclass),
                                             name=f"test_acc_byclass_rd_{self.round}")

        return test_perf

    def _train(self, rank, epoch, loader_tr, optimizer, criterion, step):
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

        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

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
            step += 1

        total_loss /= len(loader_tr)

        return step

    def train(self):
        self.imb_weights = self.generate_imbalanced_training_weights()
        os.environ['MASTER_ADDR'] = "127.0.0.1"
        os.environ['MASTER_PORT'] = str(get_free_tcp_port())
        if self.world_size > 1:
            # Prepare to launch parallel training loop
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            tmp = self.comet_experiment
            self.comet_exp_key = self.comet_experiment.get_key()
            self.comet_experiment = None

            mp.spawn(self.parallel_train_fn, args=(), nprocs=self.world_size, join=True)

            # Restore comet_exp
            self.comet_experiment = tmp
        else:
            self.parallel_train_fn(0)

    def parallel_train_fn(self, rank):
        weight_paths = self.generate_weight_paths()

        # Setup process groups and train_sampler
        train_set_wrapper = Subset(self.train_set,
                                   self.already_labeled_idxs(boolean=False, shuffle=False))

        if self.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set_wrapper,
                                                                            num_replicas=self.world_size,
                                                                            rank=rank, shuffle=True)
            dist.init_process_group("nccl", rank=rank, world_size=self.world_size)
            torch.cuda.set_device(rank)
            if rank == 0:
                self.comet_experiment = comet_ml.ExistingExperiment(
                    previous_experiment=self.comet_exp_key)

        else:
            train_sampler = None

        # Setup training data loader using train_sampler
        loader_tr_args = self.train_args['loader_tr_args'].copy()
        loader_tr_args["batch_size"] = int(loader_tr_args["batch_size"] / self.world_size)
        loader_tr = DataLoader(train_set_wrapper, shuffle=(train_sampler is None), **loader_tr_args,
                               drop_last=False, sampler=train_sampler, pin_memory=True)

        # Setup DDP network for parallel training
        self.net = self.net.to(rank)
        self.net.train()

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
            if self.freeze_feature or ('init_pretrained_ckpt_path' in self.train_args):
                self.net.eval()
            step = self._train(epoch=epoch, rank=rank, loader_tr=loader_tr, optimizer=optimizer,
                               criterion=criterion, step=step)
            scheduler.step()
            self.net = self.net.to(rank)

            # Run validation if early stop is required
            if self.validation_and_early_stopping(rank, epoch, weight_paths): break

        msg = f'Sanity Check: Best ckpt of worker rank {rank} occurs on epoch {self.best_epoch}'
        print(msg)
        self.logger.info(msg)
        self.logger.info(f"Finished training on round {self.round}")

        pass

    def validation_and_early_stopping(self, rank, epoch, weight_paths):
        if self.es_params['use_es']:
            validation_data = Subset(self.al_set, indices=self.eval_idxs)
            if self.world_size > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(validation_data,
                                                                              num_replicas=self.world_size,
                                                                              rank=rank,
                                                                              shuffle=False)
            else:
                val_sampler = None
            loader_te_args = self.train_args["loader_te_args"].copy()
            loader_te_args["batch_size"] = int(loader_te_args["batch_size"] / self.world_size)
            validation_loader = DataLoader(validation_data, shuffle=False, **loader_te_args,
                                           drop_last=False, sampler=val_sampler, pin_memory=True)

            eval_perf_dict = evaluate(validation_loader, net=self.net, metric="accuracy",
                                      num_classes=self.num_classes, net_name=self.net_name, )

            # Gather evaluation performance from all process
            if self.world_size > 1:
                eval_perf, eval_top5_acc, eval_perf_byclass = gather_parallel_eval(eval_perf_dict,
                                                                                   self.world_size,
                                                                                   self.device)
            else:
                eval_perf = eval_perf_dict['accuracy'].cpu()
                eval_top5_acc = eval_perf_dict['top_5_accuracy'].cpu()

            if self.world_size == 1 or rank == 1:
                msg = f"\tValidation performance on round {self.round} at epoch {epoch} is " \
                      f"{eval_perf * 100:.2f}%"
                self.logger.info(msg)
                # To show on stdout from within the process
                print(msg)
                self.logger.info(f"\tValidation top5 acc on round {self.round} at epoch {epoch} is "
                                 f"{eval_top5_acc * 100:.2f}%")

            if epoch % 25 == 0 and rank == 0:
                self.comet_experiment.log_metrics(
                    {f"rd_{self.round}_validation_accuracy": eval_perf,
                     f"rd_{self.round}_validation_top5_accuracy": eval_top5_acc}, step=epoch)

            # Early stopping checks
            if eval_perf >= self.es_params["best_perf"]:
                self.best_epoch = epoch
                self.es_params["count"] = 0
                self.es_params["best_perf"] = eval_perf
                if rank == 0:
                    torch.save(self.net.state_dict(), weight_paths["best_ckpt"])
            else:
                self.es_params["count"] += 1

            if self.es_params["count"] > self.es_params["patience"]:
                if rank == 0:
                    self.logger.info("Early stopping criterion reached. ")
                return True

            if rank == 0:
                torch.save(self.net.state_dict(), weight_paths["current_ckpt"])
            return False
        return False

    def generate_imbalanced_training_weights(self):
        """
        Generates weights to weigh each class differently in the loss function
        (useful if there's heavy class imbalance).

        """
        idxs_lb = self.already_labeled_idxs(boolean=False, shuffle=False)
        labels, counts = np.unique(np.array(self.train_set.targets)[idxs_lb], return_counts=True)
        weights = np.ones(shape=(self.num_classes,))
        total = counts.sum()
        for idx, l in enumerate(labels):
            weights[l] = total / counts[idx]
        weights /= weights.sum()
        return torch.Tensor(weights)

    def update(self, labeled_idxs, cur_cost):
        """
        Updates the labeled and unlabeled indices after querying.

        """
        if isinstance(labeled_idxs, list):
            labeled_idxs = np.array(labeled_idxs)
        self.idxs_lb_recent = labeled_idxs

        for idx in self.idxs_lb_recent:
            # check that examples haven't already been labeled
            assert self.idxs_lb[idx] == False
            self.idxs_lb[idx] = True

        self.cumulative_cost += cur_cost

        self.comet_experiment.log_metric(f"cumulative_budget", self.cumulative_cost,
                                         include_context=False, step=self.round)
        self.logger.info(f"Cumulative budget used on round {self.round} = {self.cumulative_cost}")
        self.comet_experiment.log_asset_data(",".join(str(e) for e in labeled_idxs),
                                             name=f"labeled_idxs_on_rd_{self.round}")
        output_dir = os.path.join(self.base_ckpt_path, self.exp_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "labeled_idxs_per_round.txt"), "a") as fh:
            fh.writelines(f"Round {self.round}: {labeled_idxs}\n")

        pass
