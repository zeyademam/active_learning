from itertools import product

def linear_evaluation_imagenet_experiments():
    dataset = "imagenet"
    dataset_dir = "<YOUR DATASET DIR HERE>"
    arg_pool = "ssp_linear_evaluation"
    model = "SSLResNet50"
    round_budget = 10000
    rounds = 8
    init_pool_size = 30000
    subset_labeled = 50000
    subset_unlabeled = 80000
    flags = "--freeze_feature --partitions 10"


    strategies = ["RandomSampler", "BalancedRandomSampler", "MASESampler",
                  "MarginSampler", "ConfidenceSampler",
                  "BASESampler",
                   "VAALSampler", "PartitionedCoresetSampler", "PartitionedBADGESampler"]

    number_of_runs = 1
    for strategy, idx in product(strategies, range(number_of_runs)):
        job_str = f"python main_al.py " \
                  f"--dataset_dir {dataset_dir} " \
                  f"--exp_name {strategy}_arg_{arg_pool}_{dataset}_b{round_budget} " \
                  f"--dataset {dataset} " \
                  f"--arg_pool {arg_pool} " \
                  f"--model {model} " \
                  f"--strategy {strategy} " \
                  f"--rounds {rounds} " \
                  f"--round_budget {round_budget} " \
                  f"--init_pool_size {init_pool_size} " \
                  f"--subset_labeled {subset_labeled} " \
                  f"--subset_unlabeled {subset_unlabeled} " \
                  f"{flags} "

        if strategy == "BalancedRandomSampler":
            job_str += f"--init_pool_type random_balance "
        else:
            job_str += f"--init_pool_type random "

        print(job_str)


def end_to_end_imagenet_experiments_pretrained():
    dataset = "imagenet"
    dataset_dir = "<YOUR DATASET DIR HERE>"
    arg_pool = "ssp_finetuning"
    model = "SSLResNet50"
    round_budget = 10000
    rounds = 8
    init_pool_size = 30000
    early_stop_patience = 30
    n_epoch = 60
    flags = "--partitions 10"
    subset_labeled = 50000
    subset_unlabeled = 80000

    strategies = ["RandomSampler", "BalancedRandomSampler", "MASESampler", "MarginSampler",
                  "ConfidenceSampler", "BASESampler",  "VAALSampler",
                  "PartitionedCoresetSampler", "PartitionedBADGESampler"]

    number_of_runs = 1
    for strategy, idx in product(strategies, range(number_of_runs)):
        job_str = f"python main_al.py " \
                  f"--dataset_dir {dataset_dir} " \
                  f"--exp_name {strategy}_arg_{arg_pool}_{dataset}_b{round_budget} " \
                  f"--dataset {dataset} " \
                  f"--arg_pool {arg_pool} " \
                  f"--model {model} " \
                  f"--strategy {strategy} " \
                  f"--rounds {rounds} " \
                  f"--round_budget {round_budget} " \
                  f"--init_pool_size {init_pool_size} " \
                  f"--early_stop_patience {early_stop_patience} " \
                  f"--n_epoch {n_epoch} " \
                  f"--subset_labeled {subset_labeled} " \
                  f"--subset_unlabeled {subset_unlabeled} " \
                  f"{flags} "

        if strategy == "BalancedRandomSampler":
            job_str += f"--init_pool_type random_balance "
        else:
            job_str += f"--init_pool_type random "

        print(job_str)


def cifar10_experiments(number_of_runs=1, n_epoch=200,
                        rounds=30, imbalanced=False,
                        round_budgets=[1000]):
    if imbalanced:
        dataset = "imbalanced_cifar10"
        imbalance_type = "exp"
        imbalance_factor = .1
        arg_pool = "ssp_finetuning_imbalanced_cifar10_imb_0_1"
    else:
        dataset = "cifar10"
        arg_pool = "ssp_finetuning"

    dataset_dir = "<YOUR DATASET DIR HERE>"

    model = "SSLResNet18"

    strategies = ["RandomSampler", "BalancedRandomSampler", "MASESampler",
              "MarginSampler", "ConfidenceSampler",
              "BASESampler", "BalancingSampler",
               "VAALSampler",  "CoresetSampler", "BADGESampler"]

    early_stop_patience = 50
    for jobid, (run_num, strategy, round_budget) in enumerate(
            product(range(number_of_runs), strategies, round_budgets)):
        init_pool_size = round_budget
        job_str = ""

        job_str += f"python main_al.py " \
                   f"--dataset_dir {dataset_dir} " \
                   f"--exp_name {strategy}_arg_{arg_pool}_{dataset}_b{round_budget} " \
                   f"--dataset {dataset} " \
                   f"--arg_pool {arg_pool} " \
                   f"--n_epoch {n_epoch} " \
                   f"--early_stop_patience {early_stop_patience} " \
                   f"--model {model} " \
                   f"--strategy {strategy} " \
                   f"--rounds {rounds} " \
                   f"--round_budget {round_budget} " \
                   f"--init_pool_size {init_pool_size} " \

        if strategy == "BalancedRandomSampler":
            job_str += f"--init_pool_type random_balance "
        else:
            job_str += f"--init_pool_type random "

        if imbalanced:
            job_str += f"--imbalance_factor {imbalance_factor} "
            job_str += f"--imbalance_type {imbalance_type} "

        print(job_str)


if __name__ == "__main__":
    linear_evaluation_imagenet_experiments()
    end_to_end_imagenet_experiments_pretrained()
    cifar10_experiments()
    cifar10_experiments(imbalanced=True)