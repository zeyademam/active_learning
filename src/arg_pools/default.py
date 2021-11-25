import os
home = os.path.expanduser("~")

dirname = os.path.dirname
args_pool = {
    "cifar10": {
        "eval_split": .01,
        "loader_tr_args": {"batch_size": 128, "num_workers": 0},
        "loader_te_args": {"batch_size": 100, "num_workers": 0},
        "optimizer": "SGD",
        "optimizer_args": {"lr": 0.1, "weight_decay": 5e-4, "momentum": 0.9},
        "lr_scheduler": "CosineAnnealingLR",
        "lr_scheduler_args": {"T_max": 200},
        "rd0_pretrained_ckpt_path": None,
    },
    "imbalanced_cifar10": {
        "eval_split": .01,
        "loader_tr_args": {"batch_size": 128, "num_workers": 0},
        "loader_te_args": {"batch_size": 100, "num_workers": 0},
        "optimizer": "SGD",
        "optimizer_args": {"lr": 0.1, "weight_decay": 5e-4, "momentum": 0.9},
        "lr_scheduler": "CosineAnnealingLR",
        "lr_scheduler_args": {"T_max": 200},
        "rd0_pretrained_ckpt_path": None,
        "imbalanced_training": True,
    },
    "imagenet": {
            "eval_split": 0.01,
            "loader_tr_args": {
                "batch_size": 128,
                "num_workers": 12,
                "prefetch_factor": 2,
            },
            "loader_te_args": {
                "batch_size": 128,
                "num_workers": 12,
                "prefetch_factor": 2,
            },
            "optimizer": "SGD",
            "optimizer_args": {"lr": 0.1, "weight_decay": 1e-4, "momentum": 0.9},
            "lr_scheduler": "StepLR",
            "lr_scheduler_args": {"step_size": 60, "gamma": 0.1},
            "rd0_pretrained_ckpt_path": None,

        },
    }
