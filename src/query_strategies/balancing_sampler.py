import numpy as np
import torch
from torch.utils.data import DataLoader

from .strategy import Strategy


class BalancingSampler(Strategy):
    """
    Implement the strategy from "Active Learning for Imbalanced Datasets" from WACV 2020
    The strategy is separated into two parts
    If the existing class distribution of labeled examples is balanced
        -> sample randomly
    If the existing class distribution of labeled examples is not balanced
        -> pick examples close to the rare class and far away from majority class
    """

    def __init__(self, train_set, al_set, net, train_args, eval_idxs, comet_experiment,
                 test_set=None, **kwargs):
        super().__init__(train_set, al_set, net, train_args, eval_idxs, comet_experiment, test_set,
                         **kwargs)
        self.saved_embeddings = None
        self.saved_ys = None
        self.freeze_feature = kwargs['freeze_feature']

    def query(self, budget):
        self.feature_net = self.net

        # Binary array where idxs_for_query[i] = True means data at index i has not been queried
        idxs_for_query = self.available_query_idxs(boolean=True)
        idxs_labeled = self.already_labeled_idxs(boolean=True)
        labeled_idxs_cur_rd = []

        if self.freeze_feature and self.saved_embeddings is not None \
                and self.saved_ys is not None:
            embeddings = self.saved_embeddings
            ys = self.saved_ys
        else:
            embedding_list = []
            y_list = []
            al_set_loader = DataLoader(self.al_set, shuffle=False,
                                       **self.train_args["loader_te_args"], drop_last=False)
            self.feature_net.eval()
            with torch.no_grad():
                for batch_idx, (x, y, idxs) in enumerate(al_set_loader):
                    x = x.to(self.device)
                    _, embedding = self.feature_net(x, return_features="finalembed")
                    embedding = embedding.cpu()
                    embedding_list.append(embedding)
                    y_list.append(y.clone())

            ys = torch.cat(y_list, dim=0)
            embeddings = torch.cat(embedding_list, dim=0)

            if self.freeze_feature:
                self.saved_embeddings = embeddings
                self.saved_ys = ys

        budget = int(min(idxs_for_query.sum(), budget))
        query_count = 0
        for _ in range(budget):
            # calculate class distribution of labeled examples
            # N: number of examples
            # N_q: number of examples available for query
            # N_l: number of labeled examples
            # C: number of classes
            # M: size of embedding
            ys_labeled = ys[idxs_labeled]  # (N_l,)
            # (1, N_l) + (C, 1) -> (C, N_l).sum(dim=1) -> (C, )
            ys_labeled_count = (ys_labeled[None, :] == torch.arange(self.num_classes)[:, None]).sum(
                dim=1)
            labeled_count = idxs_labeled.sum()
            mean_labeled_count = ys_labeled_count.float().mean()

            maj_classes = ys_labeled_count > mean_labeled_count
            maj_classes_avgcount = ys_labeled_count[maj_classes].sum() / maj_classes.sum()
            minor_classes = ys_labeled_count <= mean_labeled_count
            minor_classes_avgcount = ys_labeled_count[minor_classes].sum() / minor_classes.sum()

            # check the level of imbalance relative to the remaining budget
            # if the imbalance is high relative to the remaining budget, then use balancing strategy
            # if imbalance is is low, then use random strategy
            if budget - query_count <= minor_classes.sum() * (
                    maj_classes_avgcount - minor_classes_avgcount):
                # get embedding centers for already labeled examples
                embedding_labeled = embeddings[idxs_labeled]  # (N_l, M)
                average_matrix = torch.zeros(self.num_classes, labeled_count)  # (C, N_l)
                average_matrix[ys_labeled, torch.arange(len(ys_labeled))] = 1
                average_matrix = average_matrix / (average_matrix.sum(dim=1, keepdims=True) + 1e-5)
                # maybe we can save memory by using moving average
                centers_embedding = average_matrix @ embedding_labeled # (C, N_l) @ (N_l, M)->(C, M)
                # for classes that don't have any labeled examples, the centers_embedding would
                # be a zero vector
                centers_embedding_major = centers_embedding[maj_classes]
                rarest_class_count, rarest_class = ys_labeled_count.min(dim=0)
                centers_embedding_rarest = centers_embedding[rarest_class][None, :]

                # calculating distance between unlabeled examples and labeled class centers
                embeddings_unlabeled = embeddings[idxs_for_query]  # (N_q, M)

                a_square = embeddings_unlabeled.square().sum(dim=1, keepdims=True)
                b_square = centers_embedding_rarest.square().sum(dim=1, keepdims=True)
                ab = embeddings_unlabeled @ centers_embedding_rarest.T # (N_q, M)
                dist_to_rarest_center = a_square + b_square.T - 2*ab
                # @ (M, C_min) -> (N_q, 1)
                if rarest_class_count == 0:
                    # if the classes with the label count is zero, set the numerator of equation
                    # (9) to 1
                    dist_to_rarest_center = torch.ones_like(dist_to_rarest_center)

                a_square = embeddings_unlabeled.square().sum(dim=1, keepdims=True)
                b_square = centers_embedding_major.square().sum(dim=1, keepdims=True)
                ab = embeddings_unlabeled @ centers_embedding_major.T   # (N_q, M) @ (M, C_maj) -> (N_q, C_maj)
                dist_to_majcenters = a_square + b_square.T - 2 * ab

                min_dist_to_majcenters = dist_to_majcenters.max(dim=1,
                                                                keepdims=True).values  # (N_q, 1)
                distnorm_to_mincenters = dist_to_rarest_center / min_dist_to_majcenters  # (N_q, 1)
                distnorm_to_mincenters = distnorm_to_mincenters.squeeze()  # (N_q, )

                # selecting examples closest to the underrepresented class and farthest from the
                # overrepresented classes
                query_idx_within_unlabeledpool = distnorm_to_mincenters.min(dim=0).indices
                idxs_unlabeled = np.where(idxs_for_query == True)[0]
                query_idx = idxs_unlabeled[query_idx_within_unlabeledpool]
            else:
                # if the dataset is balance relative to the remaining query budget
                query_idx = np.random.choice(np.where(idxs_for_query.squeeze() == True)[0])

            # modifying idxs_for_query and idxs_labeled
            idxs_for_query[query_idx] = False
            idxs_labeled[query_idx] = True
            labeled_idxs_cur_rd.append(query_idx)
            query_count += 1

        return labeled_idxs_cur_rd, query_count
