import os
import logging
import scipy.io
import numpy as np
import pandas as pd
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

ROOT = "/Users/onnokampman/git_repos/variational-nnet-sbm"
DATADIR = os.path.join(ROOT, "dataset")

USER1 = 'author1'
USER2 = 'author2'
ENTRY = 'link'

N_MINIMUM_PUBLICATIONS = 9


def load_data(datadir):

    # Load raw data file.
    mat = scipy.io.loadmat(os.path.join(datadir, 'nips_1-17.mat'))

    # Get author names.
    authors_names = mat['authors_names']
    authors_names = np.concatenate(np.array(authors_names)[0, :])
    pd.DataFrame(authors_names).to_csv(os.path.join(datadir, "authors_names.csv"), index=False)

    # Extract co-publishing data.
    X = mat['docs_authors'].copy()
    X = np.dot(X.T, X)
    X = (X > 0).astype(int)
    X = X.toarray()

    # Filter out authors with few publications.
    to_keep = np.nonzero(np.sum(X, axis=0) >= N_MINIMUM_PUBLICATIONS)[0]
    X = X[to_keep, :]
    X = X[:, to_keep]
    authors_names = authors_names[to_keep]
    n_users = X.shape[1]
    pd.DataFrame(authors_names).to_csv(os.path.join(ROOT, "exp", "authors_names.csv"), index=False)

    # Convert expected format, with 3 columns: Author 1 index, Author 2 index, and the link indicator.
    il = np.tril_indices(X.shape[0], k=-1)
    data = np.vstack([il[0], il[1], X[il]]).T
    data_df = pd.DataFrame(data, columns=[USER1, USER2, ENTRY])

    return data_df, n_users, authors_names


if __name__ == "__main__":

    df, n_users, author_names = load_data(DATADIR)
    print(df.head())
