import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .strategy import Strategy


class CoresetSampler(Strategy):
    """
    Implements https://arxiv.org/abs/1708.00489
    """
    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs):
        super().__init__(train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set,
                         **kwargs)
        self.saved_pairwise_l2_dist = None
        self.freeze_feature = kwargs['freeze_feature']
        self.subset_labeled = kwargs['subset_labeled']
        self.subset_unlabeled = kwargs['subset_unlabeled']

    def get_idxs_for_coreset(self, return_sep_idxs=False):
        idxs_for_query = self.available_query_idxs(boolean=False, shuffle=True)
        idxs_labeled = self.already_labeled_idxs(boolean=False, shuffle=True)

        if self.subset_labeled is not None:
            subset_labeled = min(self.subset_labeled, len(idxs_labeled))
            idxs_labeled = idxs_labeled[:subset_labeled]
        if self.subset_unlabeled is not None:
            if self.subset_labeled is not None:
                subset_unlabeled = self.subset_labeled + self.subset_unlabeled - subset_labeled
            else:
                subset_unlabeled = self.subset_unlabeled
            subset_unlabeled = min(subset_unlabeled, len(idxs_for_query))
            idxs_for_query = idxs_for_query[:subset_unlabeled]

        # A list of indices for running coresets, this excludes indices for validation set
        idxs_for_coreset = sorted(idxs_for_query.tolist() + idxs_labeled.tolist())
        if return_sep_idxs:
            return idxs_for_coreset, idxs_labeled.tolist(), idxs_for_query.tolist()
        else:
            return idxs_for_coreset

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
    
    def get_pairwise_l2_dist(self, features):
        num_samples = features.shape[0]
        norm_square = features.square().sum(dim=1, keepdims=True).repeat((1, num_samples))
        dp = torch.mm(features, features.T)
        pairwise_l2_dist = (norm_square + norm_square.T - 2 * dp)
        return pairwise_l2_dist

    def coreset(self, pairwise_l2_dist, labeled_indicator, query_count, randomize=False):
        """

        :param
        pairwise_l2_dist : np.array shape = (N, N)
        labeled_indicator : np.array shape = (N,) a boolean that indicates examples that are labeled
        query_count : int
        :return: list of indices: list()
        """
        labeled_indicator = labeled_indicator.copy()
        labeled_idxs = []
        for _ in range(query_count):
            if labeled_indicator.sum() > 0:
                min_dist_labeled = pairwise_l2_dist[:, labeled_indicator].min(dim=1).values
                if randomize:
                    okay = False
                    min_dist_labeled = min_dist_labeled.cpu().numpy()
                    while not okay:
                        prob_selection = np.clip(min_dist_labeled, 0, None)
                        prob_selection[labeled_indicator] = 0.0
                        prob_selection = prob_selection / np.sum(prob_selection)
                        if not np.isnan(prob_selection.sum()):
                            okay = True
                        else:
                            min_dist_labeled += 0.00001

                    query_idx = np.random.choice(len(prob_selection), p=prob_selection)
                else:
                    query_idx = min_dist_labeled.max(dim=0).indices.item()
            else:
                # Choose the point where the max distance to other point is minimum
                if randomize:
                    query_idx = np.random.choice(len(labeled_indicator))
                else:
                    query_idx = pairwise_l2_dist.max(dim=1).values.min(dim=0).indices.item()
            labeled_idxs.append(query_idx)
            # Update labeled and unlabeled indices
            labeled_indicator[query_idx] = True

        return labeled_idxs

    def query(self, budget):
        # A list of indices for running coresets, this excludes indices for validation set
        idxs_for_coreset = self.get_idxs_for_coreset()

        # Pairwise l2 dist calculation
        if self.freeze_feature  and self.saved_pairwise_l2_dist is not\
                None and self.subset_unlabeled is None and self.subset_labeled is None:
            # only use saved pairwise distance if we do not changed the subset used for running
            # coreset
            pairwise_l2_dist = self.saved_pairwise_l2_dist
        else:
            embeddings = self.get_embeddings(idxs_for_coreset)
            pairwise_l2_dist = self.get_pairwise_l2_dist(embeddings)
            if self.freeze_feature:
                self.saved_pairwise_l2_dist = pairwise_l2_dist

        # idxs_labeled_bool_forcoreset is a boolean array indicating the labeled example
        # within the idxs_for_coreset subset
        idxs_labeled_bool_forcoreset = self.already_labeled_idxs(boolean=True)[idxs_for_coreset]

        budget = int(min(self.available_query_idxs(boolean=True)[idxs_for_coreset].sum(), budget))
        labeled_idxs_cur_rd = self.coreset(pairwise_l2_dist, idxs_labeled_bool_forcoreset, budget)

        # converting indices within idxs_for_coreset to the indices within the original al_set
        labeled_idxs_cur_rd = np.array(idxs_for_coreset)[labeled_idxs_cur_rd].tolist()

        return labeled_idxs_cur_rd, len(labeled_idxs_cur_rd)
