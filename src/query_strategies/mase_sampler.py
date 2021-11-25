import torch
from torch.utils.data import DataLoader, Subset
from .strategy import Strategy


class MASESampler(Strategy):
    """
    Select examples based on the distance to the decision boundary in the feature space.
    Examples closest to the decision boundaries are returned.
    Distances computed based on the last feature layer (i.e. right before linear classification
    head)
    """

    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs):
        super().__init__(train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set,
                         **kwargs)

    def query(self, budget):
        idxs_for_query = self.available_query_idxs(boolean=False, shuffle=False)
        min_margins, _, _, _ = self.compute_margins(idxs_for_query)

        budget = int(min(len(idxs_for_query), budget))
        labeled_idxs = torch.sort(min_margins, descending=False).indices[:budget]
        labeled_idxs = idxs_for_query[labeled_idxs].tolist()

        return labeled_idxs, budget

    def compute_margins(self, idxs_for_query, use_training_augmentation=False):
        if use_training_augmentation:
            al_set = Subset(self.train_set, idxs_for_query)
        else:
            al_set = Subset(self.al_set, idxs_for_query)
        al_set_loader = DataLoader(al_set, shuffle=False, **self.train_args["loader_te_args"],
                                   drop_last=False)

        radius_list = []
        margins_list = []
        pred_labels = []
        true_labels = []
        self.logger.debug(f'# batches {len(al_set_loader)}')
        self.net.eval()
        self.net = self.net.to(self.device)
        with torch.no_grad():
            if hasattr(self.net, 'module'):
                core_net = self.net.module
            else:
                core_net = self.net

            weight = core_net.linear.weight
            bias = core_net.linear.bias

            for batch_idx, (x, y, idxs) in enumerate(al_set_loader):
                x = x.to(self.device)
                logits, embedding = self.net(x, return_features= "finalembed")
                batch_size = x.size(0)
                predictions = logits.max(dim=1).indices

                # weight.shape = (C, M) C=number of classes, M = embedding size
                weight_orilogit = weight[predictions, :]
                # weight_orilogit.shape = (B, M)
                weight_delta = weight_orilogit[:, None, :] - weight[None, :]
                # weight_delta.shape = (B, C, M)
                # (B, 1, M) - (1, C, M)
                bias_delta = bias[predictions, None] - bias[None, :]
                # bias_delta.shape = (B, C)
                # (B, 1) - (1, C)
                lam_numerator = 2 * ((embedding[:, None, :] * weight_delta).sum(dim=2) + bias_delta)
                # (B, 1, M) * (B, C, M)
                # lam_numerator.shape = (B, C)
                lam_denominator = (weight_delta ** 2).sum(dim=2)
                # lam_denominator.shape = (B, C)
                lam = lam_numerator / lam_denominator
                epsilon = -weight_delta * lam[:, :, None] / 2
                # epsilon.shape = (B, C, M)
                radius = torch.linalg.norm(epsilon, dim=2)
                radius = torch.where(torch.isnan(radius), torch.tensor(float('inf')).cuda(), radius)
                # radius.shape = (B, C)
                margins, min_margins_idx = radius.min(dim=1)
                margins_list.append(margins.cpu())
                radius_list.append(radius.cpu())
                pred_labels.append(predictions.cpu())
                true_labels.append(y.cpu().clone())

            # Check the method works
            opt_epsilon = epsilon[torch.arange(batch_size), min_margins_idx]
            embedding_new = embedding + opt_epsilon
            logits_adv = self.net(embedding_new, specify_input_layer= "finalembed")
            toplogits = torch.topk(logits_adv, k=2, dim=1, largest=True)
            assert (toplogits.values[:, 0] - toplogits.values[:, 1]).abs().mean() < 0.0001

            pred_labels = torch.cat(pred_labels, dim=0)
            true_labels = torch.cat(true_labels, dim=0)
            min_margins = torch.cat(margins_list, dim=0)
            per_class_margins = torch.cat(radius_list, dim=0)
        return min_margins, per_class_margins, pred_labels, true_labels
