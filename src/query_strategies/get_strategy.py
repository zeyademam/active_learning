from .badge_sampler import BADGESampler  # noqa
from .balanced_random_sampler import BalancedRandomSampler  # noqa
from .balancing_sampler import BalancingSampler  # noqa
from .base_sampler import BASESampler  # noqa
from .confidence_sampler import ConfidenceSampler  # noqa
from .coreset_sampler import CoresetSampler  # noqa
from .margin_clustering_sampler import MarginClusteringSampler  # noqa
from .margin_sampler import MarginSampler  # noqa
from .mase_sampler import MASESampler  # noqa
from .partitioned_badge_sampler import PartitionedBADGESampler  # noqa
from .partitioned_coreset_sampler import PartitionedCoresetSampler  # noqa
from .random_sampler import RandomSampler  # noqa
from .vaal_sampler import VAALSampler  # noqa


def get_strategy(name):
    return eval(name)
