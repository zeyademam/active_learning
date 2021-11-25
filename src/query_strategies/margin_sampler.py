import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .strategy import Strategy


class MarginSampler(Strategy):
    """
    Select example based on the difference between top logit and the second largest logit.
    The difference is called output margin. Examples with the smallest margins are selected first.
    """

    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs):
        super().__init__(train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set,
                         **kwargs)

    def query(self, budget):

        idxs_for_query = self.available_query_idxs(boolean=False, shuffle=False)
        al_set = Subset(self.al_set, indices=idxs_for_query)
        al_set_loader = DataLoader(al_set, shuffle=False, **self.train_args["loader_te_args"],
                                   drop_last=False)

        output_margins_list = []
        self.logger.debug(f'# batches {len(al_set_loader)}')
        self.net.eval()
        for batch_idx, (x, y, idxs) in enumerate(al_set_loader):
            with torch.no_grad():
                x = x.to(self.device)
                logits = self.net(x)
                probs = nn.Softmax(dim=1)(logits)
                topprobs, topprobs_idx = probs.topk(dim=1, k=2, largest=True, sorted=True)
                batch_output_margins = topprobs[:, 0] - topprobs[:, 1]

            output_margins_list.append(batch_output_margins.cpu())
        self.net.train()
        output_margins = torch.cat(output_margins_list, dim=0)

        budget = int(min(len(idxs_for_query), budget))
        labeled_idxs = torch.sort(output_margins, descending=False).indices[:budget]
        labeled_idxs = idxs_for_query[labeled_idxs].tolist()

        return labeled_idxs, budget
