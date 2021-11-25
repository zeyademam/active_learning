import torch

from .mase_sampler import MASESampler


class BASESampler(MASESampler):
    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs):
        super().__init__(train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set,
                         **kwargs)

    def query(self, budget):

        idxs_for_query = self.available_query_idxs(boolean=False, shuffle=False)
        min_margins, per_class_margins, pred_labels, true_labels = self.compute_margins(
            idxs_for_query)
        budget = int(min(len(idxs_for_query), budget))

        split_labels = pred_labels

        labeled_idxs = []
        for c in range(self.num_classes):
            cur_class_query_count = int(budget / self.num_classes) + int(
                c < budget % self.num_classes)
            if cur_class_query_count == 0:
                continue

            cur_class_distance = torch.where(split_labels == c, min_margins,
                                             per_class_margins[:, c].squeeze())

            if labeled_idxs:
                cur_class_distance[labeled_idxs] = float('inf')

            cur_labeled_idxs = torch.sort(cur_class_distance, descending=False).indices
            labeled_idxs += cur_labeled_idxs[:cur_class_query_count].tolist()

        assert (len(labeled_idxs) == len(set(labeled_idxs)))

        labeled_idxs = idxs_for_query[labeled_idxs].tolist()

        return labeled_idxs, budget
