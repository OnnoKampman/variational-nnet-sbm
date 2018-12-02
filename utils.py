import numpy as np
from itertools import combinations

class BatchGenerator(object):

    def __init__(self, pairs, batch_size=None, holdout_ratio=0.1, seed=None):
        np.random.seed(seed)

        if batch_size is None:
            batch_size = len(pairs)
            print("No batch size. Using batch learning.")

        if holdout_ratio==0:
            holdout_ratio = None

        num_train = int((1.-holdout_ratio)*len(pairs)) if holdout_ratio is not None else len(pairs)
        np.random.shuffle(pairs)
        np.random.seed()
        self.train = pairs[:num_train]
        self.test = pairs[num_train:] if holdout_ratio is not None else None

        self.batch_size = batch_size
        self.num_train_batches = int(np.ceil(len(self.train) / batch_size))  # final batch may be incomplete
        self.num_test_batches = int(np.ceil(len(self.test) / batch_size)) if holdout_ratio is not None else None

        self.train_bind = 0
        self.train_idx = list(range(len(self.train)))
        self.test_bind = 0 if holdout_ratio is not None else None

        if holdout_ratio is not None:
            print("Train/test split: {}/{}".format(len(self.train), len(self.test)))

        print("Batch size {} results in {} training batches.".format(batch_size, self.num_train_batches))


    def get_idx(self, is_train):
        if is_train:
            return self.train_idx[self.train_bind*self.batch_size:\
                    (self.train_bind+1)*self.batch_size]
        else:
            return list(range(self.test_bind*self.batch_size,
                    (self.test_bind+1)*self.batch_size))

    def incr_bind(self, is_train):
        if is_train:
            self.train_bind += 1
            if (self.train_bind == self.num_train_batches):  # final batch is taken care of, slices may overflow an array
                self.train_bind = 0
                np.random.shuffle(self.train_idx)
        else:
            self.test_bind += 1
            if (self.test_bind == self.num_test_batches):
                self.test_bind = 0

    # def reset(self, is_train=True):
    #     if is_train:
    #         self.train_bind = 0
    #         np.random.shuffle(self.train_idx)
    #     else:
    #         self.test_bind = 0

    def next_batch(self, is_train=True):
        idx = self.get_idx(is_train)
        self.incr_bind(is_train)
        batch = self.train[idx] if is_train else self.test[idx]
        return batch


def log_gaussian_density(x, mu, L):

    """
    Batch log Gaussian density using the Cholesky factor of the covariance matrix.

    :param x: (..., K)-array
    :param mu: (..., K)-array
    :param L: (..., K, K)-array
    :return: (...)-array
    """

    D = x.shape[-1]
    # print("x shape:", x.shape)
    # print("mu shape:", mu.shape)
    # print("L shape:", L.shape)

    a = np.linalg.solve(L, x - mu)  # (..., K)-array

    logp = - 0.5 * D * np.log(2.0 * np.pi) - np.sum(np.log(np.diagonal(L))) \
            - 0.5 * np.sum(a**2.0, axis=-1)  # (...)-array; sums only the dimension of the Gaussian vector

    return logp


from itertools import combinations
def get_pairs(N, row, col):

    # enumerates all edges of the upper triangle
    pairs = np.array(list(combinations(range(N), 2)))
    pairs = np.column_stack((pairs, np.zeros(len(pairs), dtype=int)))
    # fill in edges
    for (r, c) in zip(row, col):
        k = r*(2*N-r-1)/2-r + c-1
        pairs[int(k), 2] = 1

    return pairs