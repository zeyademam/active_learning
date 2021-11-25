import numpy as np
import torch

from .strategy import Strategy


class BalancedRandomSampler(Strategy):
    """
    This is ONLY INTENDED AS A BASELINE STRATEGY as it cheats: it's allowed to peak at the labels
    of all samples in the train set even if they weren't queried.
    """
    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs):
        super().__init__(train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set,
                         **kwargs)

    def query(self, budget):
        """This method cheats. Query the same number of examles randomly for each class (this
        involves cheating and looking at the class labels.)


        Args:
            budget: the given budget

        Returns:

        """

        labels = torch.Tensor(self.al_set.targets)

        idxs_for_query = self.available_query_idxs(boolean=True)
        idxs_for_query = torch.Tensor(idxs_for_query).bool()
        budget = int(min(idxs_for_query.sum(), budget))

        # Calculate available examples in each class
        #   generate labels_available_count for all possible classes
        labels_for_query = labels.numpy()[idxs_for_query]
        labels_available, labels_available_count = np.unique(labels_for_query, return_counts=True)
        labels_available_count_all = np.zeros(self.num_classes)
        labels_available_count_all[labels_available.astype(np.int32)] = labels_available_count
        labels_available_count = labels_available_count_all

        #   sort labels_available_count from smallest to largest
        sort_idxs = np.argsort(labels_available_count)
        labels_count_sorted = labels_available_count[sort_idxs]
        reverse_sort_idxs = np.empty_like(sort_idxs)
        reverse_sort_idxs[sort_idxs] = np.arange(len(labels_available_count))

        #   this main algorithm increase the threshold until example count is equivalent to the total budget
        thres = budget // self.num_classes
        doneflag = False
        while doneflag is False:
            # for classes with examples smaller than and equal to the threshold, this is the most samples we could get from these classes
            count_smallerequal = np.where(labels_count_sorted <= thres, labels_count_sorted, 0).sum()
            # for classes with examples larger than the threshold, we can take either thres or thres+1 from these classes
            # we set this limitation, so the sampled classes are as balanced as possible.
            # the minimum number of examples we could sample from here is thres * (number of classes larger than thres)
            count_largermin = np.where(labels_count_sorted > thres, thres, 0).sum()
            # we then test adding 1 to classes with available samples until we reach the budget
            for oneadd in range((labels_count_sorted > thres).sum()+1):
                cum_count = count_smallerequal + count_largermin + oneadd
                if cum_count == budget:
                    doneflag = True
                    break
            if doneflag:
                break
            else:
                thres += 1
        num_classes_sample_count = np.where(labels_count_sorted <= thres, labels_count_sorted, thres)
        if oneadd > 0:
            num_classes_sample_count[-oneadd:] = thres + 1
        num_classes_sample_count = num_classes_sample_count[reverse_sort_idxs].astype(np.int32)

        assert len(num_classes_sample_count) == self.num_classes
        assert num_classes_sample_count.sum() == budget
        if not (num_classes_sample_count > labels_available_count).sum() == 0:
            import pdb
            pdb.set_trace()
        assert (num_classes_sample_count > labels_available_count).sum() == 0

        labeled_idxs = []
        for c in range(self.num_classes):
            # Compute number of samples to choose from this class
            cur_class_budget = num_classes_sample_count[c]
            if cur_class_budget == 0:
                continue

            # Figure out which idxs that belong to this class are available to choose from
            cur_available_idxs = torch.where(labels == c, idxs_for_query, False)

            # Permute the availble idxs and choose the first few up to cur_class_budget
            cur_labeled_idxs = cur_available_idxs.nonzero().squeeze()
            cur_labeled_idxs = cur_labeled_idxs[torch.randperm(cur_labeled_idxs.size(0))]
            cur_labeled_idxs = cur_labeled_idxs[:cur_class_budget]
            labeled_idxs += cur_labeled_idxs.tolist()

        assert (np.unique(labeled_idxs).shape[0] == budget)

        self.logger.info(f"Number of queried images: {budget}")

        return labeled_idxs, budget
