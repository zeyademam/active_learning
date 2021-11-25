import torch
from collections import OrderedDict


def load_pretrained_weights(net, path, replace_key=None, skip_key=None, required_key=None):
    """Loads pretrained weights from the path on disk to the network.

    This is a utility function to handle loading pretrained weights from disk
    to the network. This function handles cases when the network is initialized
    using nn.DataParallel and the weights were not or vice-versa.
    Args:
        net:
        path (str): Path to ckpt weights.
        replace_key (Optional[dict]):

    Returns:

    """
    if path is None:
        return
    if replace_key is None:
        replace_key = {}
    init_dict = net.state_dict()
    net_dict = torch.load(path)
    if 'state_dict' in net_dict:
        net_dict = net_dict['state_dict']
    net_data_parallel = ckpt_data_parallel = False
    if "module" in list(init_dict.keys())[0].lower():
        net_data_parallel = True
    if "module" in list(net_dict.keys())[0].lower():
        ckpt_data_parallel = True

    def replace_key_name(name):
        for k, v in replace_key.items():
            if k in name:
                return name.replace(k, v)
        return name
    
    def check_skip_key(k, required, skip):
        if skip:
            for s in skip:
                if s in k: return True
        
        check_res = False
        if required:
            check_res = True
            for s in required:
                if s in k: check_res = False
        return check_res

    new_state_dict = OrderedDict()
    for k, v in net_dict.items():
        if check_skip_key(k, required_key, skip_key):
            continue
        if net_data_parallel and not ckpt_data_parallel:
            new_key = "module." + replace_key_name(k)
        elif not net_data_parallel and ckpt_data_parallel:
            new_key = replace_key_name(k[7:])  # remove the prefix "module".
        else:
            new_key = replace_key_name(k)
        new_state_dict[new_key] = v

    net_dict = new_state_dict
    init_dict.update(net_dict)
    net.load_state_dict(init_dict)
    return net
