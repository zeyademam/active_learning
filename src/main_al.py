import os
import warnings
from datetime import date

import comet_ml
import cv2
import numpy as np
import torch
import torch.multiprocessing

from data_utils.top_level_data_utils import get_data
from query_strategies.get_strategy import get_strategy
from utils.generate_initial_pool import generate_eval_idxs, generate_init_lb_idxs
from utils.get_networks import get_networks
from utils.parser import get_args
from utils.resume_training import load_experiment, save_experiment
from utils.setup_logging import setup_logging
from time import time

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)
dirname = os.path.dirname

"""
Comet logging guidelines:

    used_budget: Tracks the used budget on a give round. Can be plotted vs step.
    rd_test_accuracy: Tracks the test accuracy on a given round. Can be plotted vs step. 
    budget_test_accuracy: Same as rd_test_accuracy but plotted against used budget up to that round.
        Useful when comparing experiments with different budgets. 
        Can be plotted vs step. 
    rd_{round}_train_loss: Tracks the training loss on a given round against the number of 
    gradient updates. 
        Can be plotted vs step.
    rd_{round}_validation_accuracy: Tracks the validation accuracy on a given round as a 
    function of the epoch.
        Can be plotted vs step. 
    
    
"""


def main(args):
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    # Retrieve dataset specific args
    exec(f"from arg_pools.{args.arg_pool} import args_pool as dat_arg_pool", globals())
    train_args = dat_arg_pool[args.dataset]

    # Multiple Workers # set up
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    # Prepare args for imbalance dataset
    imbalance_args = {'imbalance_type': args.imbalance_type,
                      'imbalance_factor': args.imbalance_factor,
                      'imbalance_seed': args.imbalance_seed, }

    # Parsing and preparing dataset and strategy
    train_set, test_set, al_set = get_data(data_path=args.dataset_dir, data_name=args.dataset,
                                           supervised=True, debug_mode=args.debug_mode,
                                           imbalance_args=imbalance_args, )

    # Load the network.
    device_ids = [i for i in range(torch.cuda.device_count())]
    print(f"Using {len(device_ids)} CUDA devices.")
    net = get_networks(args.dataset, args.model)
    net.freeze_feature = args.freeze_feature

    # Generate the idxs of the validation data
    eval_idxs = generate_eval_idxs(train_set, train_args["eval_split"], random_seed=99)

    # Determine initial pool size
    init_pool_size = args.init_pool_size
    if init_pool_size == -1:
        init_pool_size = int(args.round_budget)

    # Generate the labeled data at round 0
    if init_pool_size == 0:
        init_lb_idxs = np.array([])
    else:
        init_lb_idxs = generate_init_lb_idxs(train_set, eval_idxs, init_pool_size,
                                             init_pool_type=args.init_pool_type, random_seed=98)

    # If running the code in debug mode, then use fewer samples for train and
    # validation idxs.
    if args.debug_mode:
        init_lb_idxs = np.array(list(range(5)))
        if init_pool_size == 0:
            init_lb_idxs = np.array([])
        eval_idxs = np.array(list(range(15, 20)))
        test_set = torch.utils.data.Subset(test_set, list(range(10)))
    


    args.world_size = torch.cuda.device_count()

    # Fresh training logic
    if not args.resume_training:
        # Initiate new comet experiment
        experiment = comet_ml.Experiment(project_name=args.project_name, auto_param_logging=False,
                                         auto_metric_logging=False, disabled=not args.enable_comet)
        experiment.add_tag(f"{args.exp_name}")
        experiment.add_tag(f"{args.strategy}")

        # Make the experiment name unique using the comet hash
        exp_hash = os.path.basename(os.path.normpath(experiment.url))[:9]
        if exp_hash == ".":
            exp_hash = "debug"
        if not args.exp_hash:
            args.exp_hash = exp_hash

        experiment.set_name(f"{args.exp_name}")
        experiment.log_parameters(vars(args))

        # Load the Active Learning strategy.
        strategy_func = get_strategy(args.strategy)
        strategy = strategy_func(train_set, al_set, net, train_args, eval_idxs, experiment,
                                 test_set, **vars(args), )
        strategy.update(init_lb_idxs, len(init_lb_idxs))
        start_round = 0

    # Resume training setup
    else:
        strategy, start_round, experiment = load_experiment(args)

    # Torch Distributed Data Parallel (DDP) training setup
    strategy.world_size = args.world_size

    # Setup Logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    today = date.today()

    log_filename = "{}_{}{}.log".format(args.exp_hash, str(today.month).zfill(2),
                                        str(today.day).zfill(2))
    logger = setup_logging(args.log_dir, log_filename)

    logger.info(f"Experiment Name: {args.exp_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Budget used before starting: {len(init_lb_idxs)}")
    logger.info(f"Log file name: {log_filename}")
    # Start Active Learning
    for rd in range(start_round, args.rounds):
        strategy.round = rd
        logger.info(f"Active Learning Round {rd} start.")

        # Whether to use active learning querying at round 0 with pretrained SSL or transfer learned
        # checkpoint or using a randomly initialized network
        al_round_0 = rd == 0 and init_pool_size == 0

        if rd > 0 or al_round_0:
            if al_round_0:
                strategy.init_network_weights()
            labeled_idxs, cur_cost = strategy.query(args.round_budget)
            strategy.update(labeled_idxs, cur_cost)
        # We reload any SSL or Transfer learning pretrained ckpt before training at each round
        # If none were given then we randomly reset the network weights.
        init_network_weights_time_start = time()
        strategy.init_network_weights()
        init_network_weights_time = time() - init_network_weights_time_start
        print(f'Rd {rd} init_network_weights_time is {init_network_weights_time}')

        train_time_start = time()
        strategy.train()
        train_time = time() - train_time_start
        print(f'Rd {rd} train_time is {train_time}')

        load_best_ckpt_time_start = time()
        strategy.load_best_ckpt()
        load_best_ckpt_time = time() - load_best_ckpt_time_start
        print(f'Rd {rd} load_best_ckpt_time is {load_best_ckpt_time}')

        test_time_start = time()
        strategy.test()
        test_time = time() - test_time_start
        print(f'Rd {rd} test_time is {test_time}')

        save_experiment(strategy, args, logger)
        args.resume_training = True
        if len(strategy.available_query_idxs()) == 0:
            logger.info("Finished querying all Images!")
            break

if __name__ == "__main__":
    args = get_args()
    main(args)
