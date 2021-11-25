import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .strategy import Strategy
from .coreset_sampler import CoresetSampler


class PartitionedCoresetSampler(CoresetSampler):
    """
    Inspired by the implementation in https://arxiv.org/abs/2107.14263
    """
    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs):
        super().__init__(train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set,
                         **kwargs)
        self.partitions = kwargs['partitions']

    def get_embeddings(self, idxs):
        al_set = Subset(self.al_set, indices=idxs)
        al_set_loader = DataLoader(al_set, shuffle=False, **self.train_args["loader_te_args"],
                                   drop_last=False)
        embedding_list = []
        self.logger.debug(f'# batches {len(al_set_loader)}')
        self.feature_net.eval()
        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(al_set_loader):
                x = x.to(self.device)
                _, embedding = self.feature_net(x, return_features="finalembed")
                embedding = embedding.cpu()
                embedding_list.append(embedding)
        embeddings = torch.cat(embedding_list, dim=0)
        return embeddings

    # Random partition given idxs into <self.paritions> part
    def generate_partition_idxs_list(self, input_idxs):
 
        partition_idxs_list = []
        cum_idx = 0
        idxs = np.array(input_idxs)
        np.random.shuffle(idxs)
        for i in range(self.partitions):
            idxs_num = len(input_idxs)
            cur_range = int(idxs_num / self.partitions) + int(i < idxs_num % self.partitions)
            partition_idxs_list.append(idxs[cum_idx:cum_idx+cur_range])
            cum_idx += cur_range
        return partition_idxs_list
    
    def query(self, budget):
        return self._query_with_embedding_func(budget, self.get_embeddings)

    def _query_with_embedding_func(self, budget, embed_f, randomize_coreset=False):
        # A list of labeled indices and a list of unlabeld indices for running coresets. 
        # Having separated lists to ensure each random partition has the same # of labeled
        # and unlabeled examples.
        # This excludes indices for validation set.
        _, labeled_idxs, unlabeled_idxs = self.get_idxs_for_coreset(return_sep_idxs=True)
        labeled_idxs_partition_list = self.generate_partition_idxs_list(labeled_idxs)
        unlabeled_idxs_partition_list = self.generate_partition_idxs_list(unlabeled_idxs)

        budget = int(min(len(unlabeled_idxs), budget))
        labeled_idxs_cur_rd = []  
        for i in range(self.partitions):
            # Prepare idxs for current partition
            labeled_idxs_partition = labeled_idxs_partition_list[i]
            unlabeled_idxs_partition = unlabeled_idxs_partition_list[i]
            idxs_partition = np.concatenate((labeled_idxs_partition, unlabeled_idxs_partition))

            embeddings = embed_f(idxs_partition)
            pairwise_l2_dist = self.get_pairwise_l2_dist(embeddings)
            
            cur_budget = int(budget / self.partitions) + int(i < budget % self.partitions)
            # The first <len(labeled_idxs_partition)> idxs of idxs_partition are labeled
            labeled_indicator = np.array([False] * len(idxs_partition))
            labeled_indicator[:len(labeled_idxs_partition)] = True

            new_labeled_idxs = self.coreset(pairwise_l2_dist, labeled_indicator, cur_budget, randomize=randomize_coreset)
            # translate idxs from local partiton view to global view
            new_labeled_idxs = idxs_partition[new_labeled_idxs]
            labeled_idxs_cur_rd += list(new_labeled_idxs)

        assert (len(labeled_idxs) == len(set(labeled_idxs)))

        return sorted(labeled_idxs_cur_rd), len(labeled_idxs_cur_rd)
