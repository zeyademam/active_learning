import os

dirname = os.path.dirname
args_pool = {
    "imagenet": {
        "eval_split": 0.01,
        "loader_tr_args": {
            "batch_size": 128,
            "num_workers": 8,
            "prefetch_factor": 2,
        },
        "loader_te_args": {
            "batch_size": 128,
            "num_workers": 8,
            "prefetch_factor": 2,
        },
        "optimizer": "SGD",
        "optimizer_args": {"lr": 15, "weight_decay": 1e-4, "momentum": 0.9},
        "lr_scheduler": "StepLR",
        "lr_scheduler_args": {"step_size": 20, "gamma": 0.1},
        "init_pretrained_ckpt_path": "../pretrained_ckpt/imagenet/moco_v2_800ep_pretrain.pth.tar",
        "required_key": ["encoder_q"],
        "skip_key": ["fc"],
        "replace_key": {"encoder_q": "encoder"},
    }
}

