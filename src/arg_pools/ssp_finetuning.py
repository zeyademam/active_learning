import os

dirname = os.path.dirname
args_pool = {
    "cifar10": {
        "eval_split": 0.1,
        "loader_tr_args": {"batch_size": 128, "num_workers": 2},
        "loader_te_args": {"batch_size": 100, "num_workers": 2},
        "optimizer": "SGD",
        "optimizer_args": {"lr": 0.001, "weight_decay": 5e-4, "momentum": 0.9},
        "lr_scheduler": "CosineAnnealingLR",
        "lr_scheduler_args": {"T_max": 200},
        "init_pretrained_ckpt_path": "../pretrained_ckpt/cifar10/simclr.pth.tar",
        "required_key": ["encoder"],
        "skip_key": ["linear"],
        "replace_key": None,
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
        "optimizer_args": {"lr": 0.001, "weight_decay": 0, "momentum": 0.9},
        "lr_scheduler": "StepLR",
        "lr_scheduler_args": {"step_size": 10, "gamma": 0.1},
        "init_pretrained_ckpt_path": "../pretrained_ckpt/imagenet/moco_v2_800ep_pretrain.pth.tar",
        "required_key": ["encoder_q"],
        "skip_key": ["fc"],
        "replace_key": {"encoder_q": "encoder"},
    },
}
