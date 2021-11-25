import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .strategy import Strategy


class ConfidenceSampler(Strategy):
    """
    Select examples with the smallest top softmaxed logit.
    """

    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs):
        super().__init__(train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set,
                         **kwargs)

    def query(self, budget):
        idxs_for_query = self.available_query_idxs()
        al_set = Subset(self.al_set, indices=idxs_for_query)
        al_set_loader = DataLoader(al_set, shuffle=False, **self.train_args["loader_te_args"],
                                   drop_last=False)

        confidence_list = []
        self.logger.debug(f'# batches {len(al_set_loader)}')
        self.net.eval()
        for batch_idx, (x, y, idxs) in enumerate(al_set_loader):
            with torch.no_grad():
                x = x.to(self.device)
                logits = self.net(x)
                probs = nn.Softmax(dim=1)(logits)
                topprobs, topprobs_idx = probs.topk(dim=1, k=1, largest=True, sorted=True)
                confidence_b = topprobs[:, 0]

            confidence_b = confidence_b.cpu()
            confidence_list.append(confidence_b)
        self.net.train()
        confidence = torch.cat(confidence_list, dim=0)

        query_count = int(min(len(idxs_for_query), budget))
        confidence = confidence[idxs_for_query]
        labeled_idxs = torch.sort(confidence, descending=False).indices
        labeled_idxs = labeled_idxs[:query_count]
        labeled_idxs = idxs_for_query[labeled_idxs]
        labeled_idxs = labeled_idxs.tolist()

        return labeled_idxs, query_count
