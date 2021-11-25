import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .coreset_sampler import CoresetSampler

POOLING_H = 16
POOLING_AREA = 512


class BADGESampler(CoresetSampler):
    """
    Based on https://arxiv.org/abs/1906.03671
    """
    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs):
        super().__init__(train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set,
                         **kwargs)

    def get_gradient_embeddings(self, idxs, use_adaptive_pool=False):
        al_set = Subset(self.al_set, indices=idxs)
        al_set_loader = DataLoader(al_set, shuffle=False, **self.train_args["loader_te_args"],
                                   drop_last=False)
        embedding_list = []
        self.logger.debug(f'# batches {len(al_set_loader)}')
        self.feature_net.eval()
        for batch_idx, (x, y, idxs) in enumerate(al_set_loader):
            with torch.no_grad():
                x = x.to(self.device)
                logits, embedding = self.net(x, return_features="finalembed")
                max_logit, max_logit_idx = logits.max(dim=1)

            logits.requires_grad = True
            loss = nn.CrossEntropyLoss()(logits, max_logit_idx)
            grad = torch.autograd.grad(loss, logits)[0]
            # manually calculate gradient with respect to each embeddings
            with torch.no_grad():
                grad_embed = grad[:, :, None] * embedding[:, None, :]
                if use_adaptive_pool:
                    pool_h = min(POOLING_H, grad_embed.size(1))
                    pool_w = int(float(POOLING_AREA) / pool_h)
                    grad_embed = F.adaptive_avg_pool2d(grad_embed, (pool_h, pool_w))
                grad_embed = grad_embed.view(grad_embed.size(0), -1).cpu()
            embedding_list.append(grad_embed)
        embeddings = torch.cat(embedding_list, dim=0)
        return embeddings

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
            embeddings = self.get_gradient_embeddings(idxs_for_coreset)
            num_samples = embeddings.shape[0]
            norm_square = embeddings.square().sum(dim=1, keepdims=True).repeat((1, num_samples))
            dp = torch.mm(embeddings, embeddings.T)
            pairwise_l2_dist = (norm_square + norm_square.T - 2 * dp)

        # idxs_labeled_bool_forcoreset is a boolean array indicating the labeled example
        # within the idxs_for_coreset subset
        idxs_labeled_bool_forcoreset = self.already_labeled_idxs(boolean=True)[idxs_for_coreset]

        budget = int(min(self.available_query_idxs(boolean=True)[idxs_for_coreset].sum(), budget))
        labeled_idxs_cur_rd = self.coreset(pairwise_l2_dist, idxs_labeled_bool_forcoreset, budget,
                                           randomize=True)

        # converting indices within idxs_for_coreset to the indices within the original al_set
        labeled_idxs_cur_rd = np.array(idxs_for_coreset)[labeled_idxs_cur_rd].tolist()

        return labeled_idxs_cur_rd, len(labeled_idxs_cur_rd)
