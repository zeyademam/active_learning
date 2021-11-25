import argparse

ckpt_path = "../checkpoint"
log_dir = "./logs"


def get_args():
    """
    Parse the terminal args.

    """
    parser = argparse.ArgumentParser(description="")

    # Comet Experiment and Logging
    parser.add_argument("--project_name", dest="project_name", default="active-learning", type=str,
                        help="project name the experiment")
    parser.add_argument("--exp_name", dest="exp_name", default="active_learning", type=str,
                        help="exp_name for specification")
    parser.add_argument("--log_dir", dest="log_dir", default=log_dir, help="logs are saved here")
    parser.add_argument("--enable_comet", dest="enable_comet", action="store_true",
                        help="Whether to enable comet ML logging.")

    # Dataset Arguments
    parser.add_argument("--dataset", dest="dataset", default="cifar10", type=str,
                        help="path of the dataset")
    parser.add_argument("--dataset_dir", dest="dataset_dir",
                        help="path to the root dir of datasets")
    parser.add_argument("--arg_pool", dest="arg_pool", default="default",
                        help="Dataset specific args to use for this AL experiment")

    # Imbalance Dataset Arguments
    parser.add_argument("--imbalance_type", dest='imbalance_type', default=None,
                        choices=['exp', 'step'],
                        help='Imbalance type. Step means c/2 maj classes and c/2 minor classes. '
                             'Exp means # of examples decay exponentially')
    parser.add_argument("--imbalance_factor", dest='imbalance_factor', default=0.1, type=float,
                        help='Imbalance factor.')
    parser.add_argument("--imbalance_seed", dest='imbalance_seed', default=0, type=int,
                        help='Imbalance random seed for generating imbalanced dataset.')

    # Global active learning parameters
    parser.add_argument("--strategy", dest="strategy", default="RandomSampler",
                        help="strategy for active learning")
    parser.add_argument("--rounds", dest="rounds", type=int, default=5,
                        help="# of rounds to do active learning")
    parser.add_argument("--round_budget", dest="round_budget", type=float, default=5000,
                        help="Budget to exhaust per round.")
    parser.add_argument("--freeze_feature", dest='freeze_feature', default=False,
                        action='store_true',
                        help="Used to train the final linear layer while keeping the backbone of "
                             "the model fixed")
    parser.add_argument("--init_pool_size", dest='init_pool_size', type=int, default=-1)
    parser.add_argument("--init_pool_type", dest='init_pool_type', type=str, default='random',
                        choices=['random', 'random_balance'])

    # Global training args
    parser.add_argument("--model", dest="model", default="SSLResNet18", type=str)
    parser.add_argument("--resume_training", dest="resume_training", action="store_true")
    parser.add_argument("--exp_hash", dest="exp_hash", default=None, type=str)
    parser.add_argument("--ckpt_path", dest="ckpt_path", type=str, default=ckpt_path)
    parser.add_argument("--n_epoch", dest="n_epoch", type=int, default=60,
                        help="The number of training epochs.")
    parser.add_argument("--early_stop_patience", dest="early_stop_patience", type=int, default=30,
                        help="Early stopping patience. "
                             "If validation jaccard has not reached a new best"
                             "value in N epochs, the training stops. If 0,"
                             "then early stopping will not be used.", )

    # Debugging params
    parser.add_argument("--debug_mode", dest="debug_mode", default=False, action="store_true",
                        help="Use debug mode")

    # Partitioned Coreset and Partitioned BADGE Argument
    parser.add_argument("--subset_labeled", dest='subset_labeled', type=int,
                        help='The number of labeled subset used for running coreset. ')
    parser.add_argument("--subset_unlabeled", dest='subset_unlabeled', type=int,
                        help='The number of unlabeled subset used for running coreset. ')
    parser.add_argument("--partitions", dest='partitions', type=int, default=1,
                        help='The number of random partitions to perform coreset separately on.')

    # VAAL arguments
    parser.add_argument("--vae_latent_dim", dest="vae_latent_dim", type=int, default=64,
                        help="Imagenet 64 and CIFAR10 32")
    parser.add_argument("--vaal_adversary_param", dest="vaal_adversary_param", type=float,
                        default=10.,
                        help="This is lambda2 in the VAAL paper, 10 for imagenet and 1 for cifar10")
    parser.add_argument("--lr_vae", dest="lr_vae", type=float, default=5e-5,
                        help="ImageNet 5e-5, CIFAR 5e-4")
    parser.add_argument("--lr_discriminator", dest="lr_discriminator", type=float, default=1e-3,
                        help="ImageNet 1e-3, CIFAR 5e-4")

    return parser.parse_args()
