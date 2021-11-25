import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import AgglomerativeClustering
from .strategy import Strategy
import torch.nn as nn


class MarginClusteringSampler(Strategy):
    """
    Based on https://arxiv.org/abs/2107.14263
    """
    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs):
        super().__init__(train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set,
                         **kwargs)
        self.saved_pairwise_l2_dist = None
        self.freeze_feature = kwargs['freeze_feature']
        self.subset_labeled = kwargs['subset_labeled']
        self.subset_unlabeled = kwargs['subset_unlabeled']
        self.cluster_assignment = None

    def get_embeddings_and_margins(self, idxs):
        al_set = Subset(self.al_set, indices=idxs)
        al_set_loader = DataLoader(al_set, shuffle=False, **self.train_args["loader_te_args"],
                                   drop_last=False)
        embedding_list = []
        output_margins_list = []
        self.logger.debug(f'# batches {len(al_set_loader)}')
        self.feature_net.eval()
        with torch.no_grad():
            for batch_idx, (x, y, idxs) in enumerate(al_set_loader):
                x = x.to(self.device)
                logits, embedding = self.feature_net(x, return_features="finalembed")
                probs = nn.Softmax(dim=1)(logits)
                topprobs, topprobs_idx = probs.topk(dim=1, k=2, largest=True, sorted=True)
                batch_output_margins = topprobs[:, 0] - topprobs[:, 1]
                embedding = embedding.cpu()
                embedding_list.append(embedding)
                batch_output_margins = batch_output_margins.cpu()
                output_margins_list.append(batch_output_margins)
        embeddings = torch.cat(embedding_list, dim=0)
        output_margins = torch.cat(output_margins_list, dim=0)
        return embeddings, output_margins


    def query(self, budget):
        # Run HAC on unlabeled examples only
        if self.subset_unlabeled is None:
            idxs_for_HAC = self.available_query_idxs(boolean=False, shuffle=False)
        else:
            idxs_for_HAC = sorted(
                self.available_query_idxs(boolean=False, shuffle=True)[:self.subset_unlabeled]
            )
        embeddings, output_margins = self.get_embeddings_and_margins(idxs_for_HAC)
        if self.cluster_assignment is None or self.subset_unlabeled:
            # only do clustering in the first round or
            # redo clustering in every round when random subset is used
            cluster_assignment = AgglomerativeClustering(n_clusters=20).fit(embeddings).labels_
        else:
            cluster_assignment = self.cluster_assignment
        # ex: [0, 0, 0, 2, 2, 1]

        cluster_ids, cluster_count = np.unique(cluster_assignment, return_counts=True)
        # cluster ids sorted by the size of the clusters starting from small to large
        cluster_ids_sorted = [id for _, id in sorted(zip(cluster_count, cluster_ids))]
        query_idxs = []
        query_count = 0
        start_cluster = 0
        budget = int(min(len(idxs_for_HAC), budget))
        while query_count < budget:
            #round robin sampling
            for i in range(start_cluster, len(cluster_count)):
                current_cluster_ids = cluster_ids_sorted[i]
                margins_current_cluster = output_margins[cluster_assignment==current_cluster_ids]
                min_margin_idx_incluster = np.argmin(margins_current_cluster)

                potential_query_indices = np.where(cluster_assignment==current_cluster_ids)[0]
                query_idx_inunlabeled = potential_query_indices[min_margin_idx_incluster]
                cluster_assignment[query_idx_inunlabeled] = -1
                query_idx_inalset = idxs_for_HAC[query_idx_inunlabeled]
                query_idxs.append(query_idx_inalset)
                query_count += 1
                if len(margins_current_cluster) == 1:
                    start_cluster += 1
                if query_count >= budget:
                    break

        self.cluster_assignment = cluster_assignment[cluster_assignment!=-1]
        return query_idxs, budget
