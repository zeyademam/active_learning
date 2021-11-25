import numpy as np
import torch
from .strategy import Strategy


class RandomSampler(Strategy):
    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set=None, **kwargs):
        super(RandomSampler, self).__init__(
            train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set, **kwargs
        )

    def query(self, budget):
        """Query full images based until the given budget is exhausted.

        Args:
            budget: the given budget

        Returns:

        """
        idxs_for_query = self.available_query_idxs()
        labeled_idxs = []

        count = 0
        for count, idx in enumerate(idxs_for_query):
            # We can query this mask add it to the candidate list
            labeled_idxs.append(idx)
            if count + 1 == budget:
                break

        self.logger.info(f"Number of queried images: {count + 1}")

        return labeled_idxs, count+1
