import logging

import numpy as np

logger = logging.getLogger("ActiveLearning")


def generate_idxs(train_set, size, generation_type: str, avoid_idxs=None, random_seed=None):
    rng = np.random.default_rng(random_seed)
    available_idxs = np.arange(len(train_set))
    if avoid_idxs is not None:
        available_idxs = np.setdiff1d(available_idxs, avoid_idxs)

    if generation_type == 'random':
        rng.shuffle(available_idxs)
        return available_idxs[:size]
    elif generation_type == 'random_balance':
        num_classes = train_set.num_classes
        # Make sure each class has the same # of labeled samples
        if not size % num_classes == 0:
            size = size - size % num_classes
            print(f"The size of the data was reduced to {size} in order to obtain a "
                  f"balanced dataset")

        # Calculate available examples in each class
        targets = np.array(train_set.targets)[available_idxs]
        labels_available, labels_available_count = np.unique(targets, return_counts=True)
        sort_idxs = np.argsort(labels_available_count)
        labels_count_sorted = labels_available_count[sort_idxs]
        reverse_sort_idxs = np.arange(len(labels_available_count))[sort_idxs]

        thres = size // num_classes
        doneflag = False
        while doneflag is False:
            count_smallerequal = np.where(labels_count_sorted <= thres, labels_count_sorted,
                                          0).sum()
            count_largermin = np.where(labels_count_sorted > thres, thres, 0).sum()
            for oneadd in range((labels_count_sorted > thres).sum()):
                cum_count = count_smallerequal + count_largermin + oneadd
                if cum_count == size:
                    doneflag = True
                    break
            if doneflag:
                break
            else:
                thres += 1
        num_classes_sample_count = np.where(labels_count_sorted <= thres, labels_count_sorted,
                                            thres)
        if oneadd > 0:
            num_classes_sample_count[-oneadd:] = thres + 1
        num_classes_sample_count = num_classes_sample_count[reverse_sort_idxs]

        assert len(num_classes_sample_count) == num_classes
        assert num_classes_sample_count.sum() == size
        assert (num_classes_sample_count > labels_available_count).sum() == 0

        result = []
        rng.shuffle(available_idxs)
        for idx in available_idxs:
            if size == 0:
                break
            y = train_set.targets[idx]
            if num_classes_sample_count[y] > 0:
                result.append(idx)
                num_classes_sample_count[y] -= 1
                size -= 1
        return np.array(result)
    else:
        raise ValueError('Init pool type not implemented')


def generate_eval_idxs(train_set, ratio=0.1, random_seed=None):
    eval_size = int(len(train_set) * ratio)
    return generate_idxs(train_set=train_set, size=eval_size, random_seed=random_seed,
                         generation_type="random_balance")


def generate_init_lb_idxs(train_set, eval_idxs, init_pool_size, init_pool_type, random_seed=None):
    return generate_idxs(train_set=train_set, size=init_pool_size, generation_type=init_pool_type,
                         random_seed=random_seed, avoid_idxs=eval_idxs)
