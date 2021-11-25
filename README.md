# Active Learning at the ImageNet Scale

This repo contains code for the paper _Active Learning at the ImageNet Scale_ by 
_Zeyad Emam*, Hong-Min Chu*, Ping-Yeh Chiang*, Wojtek Czaja, Richard Leapman, Micah Goldblum, 
and Tom Goldstein._

## Requirements

`pip install -r requirements.txt`

## Comet and Logging

This project uses Comet ML to log all experiments, you must
install [comet_ml](https://www.comet.ml) (included in requirements.txt), however, the code does not
require the user to have a Comet ML account or to enable comet logging at all. If you choose to use
comet ML, then you should include your API key in your home directory
`~/.comet.config` (more on this in the Comet ML documentation). To use comet make sure the use the
flag `--enable_comet`.

Logs and network weights are stored according to the command line arguments `--log_dir`
and `--ckpt_path`.

## Loading SSP checkpoints

Self-supervised pretrained checkpoints must be obtained separately and specified
in `./src/arg_pools` for each argpool, under the key `"init_pretrained_ckpt_path"`.
To access the checkpoints used in our experiments, please use the following links:
- [ResNet-18 checkpoint for CIFAR-10](https://drive.google.com/file/d/1jN0A9SDj_bvwyDGPwvPPvcc-iIpfdEJf/view?usp=sharing)
- [ResNet-18 checkpoint for imbalanced CIFAR-10](https://drive.google.com/file/d/1QzJV0C4kkGqXNPkn6ifySEFKueKBiwXi/view?usp=sharing)
- [ResNet-50 checkpoint for ImageNet](https://drive.google.com/file/d/17px0_0syO3QNmuQGlLQuw00rvSG3ypuH/view?usp=sharing)


## Sample Commands to Reproduce the Results in the Paper

Each Imagenet experiment was conducted on a cluster node with a single V100-SXM2 GPU (32GB VRAM),
64gb of RAM, and 16 2.3 GHz Intel Gold 6140 cpus. If more than one gpu are available on the node,
the code will automatically distribute batches across all gpus using DistributedDataParallel
training.

Below is a sample command for running an experiment. The full list of command line arguments can be
found in `src/utils/parser.py`.

```
python main_al.py --dataset_dir <YOUR DATASET DIR HERE> --exp_name RandomSampler_arg_ssp_linear_evaluation_imagenet_b10000 --dataset imagenet --arg_pool ssp_linear_evaluation --model SSLResNet50 --strategy RandomSampler --rounds 8 --round_budget 10000 --init_pool_size 30000 --subset_labeled 50000 --subset_unlabeled 80000 --freeze_feature --partitions 10 --init_pool_type random 
```

The full list of commands to reproduce all plots in the paper can be obtained by
running `python src/gen_jobs.py`.



