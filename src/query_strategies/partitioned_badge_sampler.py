from .badge_sampler import BADGESampler
from .partitioned_coreset_sampler import PartitionedCoresetSampler


class PartitionedBADGESampler(BADGESampler, PartitionedCoresetSampler):
    """
    Inspired by the implementation in https://arxiv.org/abs/2107.14263
    """
    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs):
        super().__init__(train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set,
                         **kwargs)

    def get_gradient_embeddings_with_pooling(self, idxs):
        return self.get_gradient_embeddings(idxs, use_adaptive_pool=True)

    def query(self, budget):
        return self._query_with_embedding_func(budget, self.get_gradient_embeddings_with_pooling,
                                               randomize_coreset=True)
