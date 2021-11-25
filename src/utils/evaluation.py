import logging

import torch
import torch.distributed as dist
from torch.autograd import Variable

SMOOTH = 1e-6
logger = logging.getLogger("ActiveLearning")


def accuracy(dataloader, net, top_k=(1, 5), **kwargs):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net_name = kwargs["net_name"]
    if "weights_path" in kwargs.keys():
        net_dict = torch.load(kwargs["weights_path"], map_location=device)
        net.load_state_dict(net_dict)
        del net_dict
    net.eval()

    sample_input, sample_output, _ = next(iter(dataloader))

    max_k = max(top_k)
    corrects_dict = {k: 0.0 for k in top_k}
    if "dataset" in dataloader.dataset.__dict__:
        num_classes = dataloader.dataset.dataset.num_classes
    else:
        num_classes = dataloader.dataset.num_classes
    corrects_byclass = torch.zeros((num_classes,), dtype=torch.float)  # need to change this by
    # class count dynamically
    count_byclass = torch.zeros((num_classes,), dtype=torch.float)
    for batch_idx, data in enumerate(dataloader):
        inputs, targets, idxs = data
        output_ = []
        with torch.no_grad():
            inputs = inputs.to(device)  # [N, C, H, W]
            targets = targets.to(device)  # [N,]
            output = net(inputs)  # [N, O]

            _, predictions = torch.topk(output, max_k, dim=1, largest=True, sorted=True)
            for k in top_k:
                k_pred = predictions[:, :k]
                corrects = torch.sum(k_pred == targets.unsqueeze(1).expand_as(k_pred)).to('cpu')
                corrects_dict[k] += corrects
                if k == 1:
                    corrects_vec = (k_pred == targets[:, None]).to('cpu')  # [N, 1]
                    corrects_byclass_b = corrects_vec & (
                            targets.to('cpu')[:, None] == torch.arange(num_classes)[None, :])
                    corrects_byclass += corrects_byclass_b.sum(dim=0).float()
                    count_byclass += (targets.to('cpu')[:, None] == torch.arange(num_classes)[None,
                                                                    :]).sum(dim=0).float()

            msg = f"\tEval Batch {batch_idx + 1}/{len(dataloader)}"
            if batch_idx % 25 == 0:
                logger.info(msg)

    output = {}
    for k in corrects_dict:
        output[f'top_{k}_correct_count'] = corrects_dict[k]
        output[f'top_{k}_accuracy'] = corrects_dict[k] / len(dataloader.dataset)
    output['accuracy'] = corrects_dict[1] / len(dataloader.dataset)
    output['accuracy_byclass'] = corrects_byclass / count_byclass
    output['count_byclass'] = count_byclass
    output['corrects_byclass'] = corrects_byclass
    output['count'] = len(dataloader.dataset)
    return output


def gather_parallel_eval(eval_dict, world_size, device):
    assert world_size > 1

    def gather_variable_int(name):
        org_var = eval_dict[name]
        org_var = torch.Tensor([org_var])
        org_var = org_var.to(device)
        arrs = [torch.zeros_like(org_var) for _ in range(world_size)]
        dist.all_gather(arrs, org_var)
        return sum(arrs).item()

    def gather_variable_tensor(name):
        org_var = eval_dict[name].clone()
        org_var = org_var.to(device)
        arrs = [torch.zeros_like(org_var) for _ in range(world_size)]
        dist.all_gather(arrs, org_var)
        return sum(arrs)

    # Gather number of examples
    count = gather_variable_int('count')
    count_byclass = gather_variable_tensor('count_byclass')
    # Gather number of correct predictions
    top_1_correct_count = gather_variable_int('top_1_correct_count')
    top_5_correct_count = gather_variable_int('top_5_correct_count')
    corrects_byclass = gather_variable_tensor('corrects_byclass')
    top_1_acc = top_1_correct_count / count
    top_5_acc = top_5_correct_count / count
    accuracy_byclass = corrects_byclass / count_byclass

    return top_1_acc, top_5_acc, accuracy_byclass.cpu()


def evaluate(dataloader, **kwargs):
    metric = kwargs["metric"]
    metric_func = eval(metric)

    return metric_func(dataloader, **kwargs)
