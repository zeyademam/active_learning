import os

dirname = os.path.dirname
args_pool = {
    "imbalanced_cifar10": {
        "eval_split": 0.1,
        "loader_tr_args": {"batch_size": 128, "num_workers": 2},
        "loader_te_args": {"batch_size": 100, "num_workers": 2},
        "optimizer": "SGD",
        "optimizer_args": {"lr": 0.002, "weight_decay": 0, "momentum": 0.9},
        "lr_scheduler": "CosineAnnealingLR",
        "lr_scheduler_args": {"T_max": 200},
        "init_pretrained_ckpt_path": "../pretrained_ckpt/cifar10/simclr_imb_pretrain0_1.tar",
        "required_key": ["encoder"],
        "skip_key": ["linear"],
        "replace_key": None,
        "imbalanced_training": True,
    },
}

