import logging
import os
import pickle

import comet_ml


def load_experiment(args, check_args_match=True):
    assert args.exp_hash
    exp_ckpt_prefix = f"{args.ckpt_path}/{args.exp_name}_{args.exp_hash}"
    if not os.path.exists(exp_ckpt_prefix):
        raise ValueError("check point does not exist")
    prev_args = pickle.load(open(f"{exp_ckpt_prefix}/args.pick", "rb"))
    # Check args match
    cur_args_dict = vars(args)
    prev_args_dict = vars(prev_args)
    # TODO: Fix this in the future
    tmp1 = {key: val for key, val in cur_args_dict.items() if
            key not in {"resume_training", "exp_name", "world_size"}}
    tmp2 = {key: val for key, val in prev_args_dict.items() if
            key not in {"resume_training", "exp_name", "world_size"}}
    if check_args_match and tmp1 != tmp2:
        logging.warning("Loaded experiment however args are not the same!")
        logging.warning(f"Initial args: {prev_args}")
        logging.warning(f"Current args: {args}")

    prev_status = pickle.load(open(f"{exp_ckpt_prefix}/status.pick", "rb"))
    prev_strategy = pickle.load(open(f"{exp_ckpt_prefix}/strategy.pick", "rb"))
    prev_experiment = comet_ml.ExistingExperiment(previous_experiment=prev_status["comet_exp_key"])
    prev_experiment.add_tag(f"{args.exp_name}")
    prev_experiment.add_tag(f"{args.strategy}")
    prev_strategy.comet_experiment = prev_experiment
    prev_round = prev_status["round"]

    return prev_strategy, prev_round + 1, prev_experiment


def save_experiment(strategy, args, logger):
    exp_ckpt_prefix = f"{args.ckpt_path}/{args.exp_name}_{args.exp_hash}"
    if not os.path.exists(exp_ckpt_prefix):
        mode = 0o777
        os.mkdir(exp_ckpt_prefix, mode)
    status_dict = dict()
    status_dict["round"] = strategy.round
    status_dict["comet_exp_key"] = strategy.comet_experiment.get_key()
    tmp_exp = strategy.comet_experiment
    strategy.comet_experiment = None
    pickle.dump(status_dict, open(f"{exp_ckpt_prefix}/status.pick", "wb"))
    pickle.dump(strategy, open(f"{exp_ckpt_prefix}/strategy.pick", "wb"))
    pickle.dump(args, open(f"{exp_ckpt_prefix}/args.pick", "wb"))
    strategy.comet_experiment = tmp_exp
    logger.info(f"Save experiment {args.exp_name} at round {strategy.round}")
