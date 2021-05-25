import numpy as np

import numpy.linalg as LA

from scipy import sparse

from sklearn.utils import gen_batches





def pca_fit(X, var_ratio=1, return_transform=True):

    """



    Parameters

    ----------

    X : array_like

        An array of data samples with shape (n_samples, n_features).

    var_ratio : float

        The variance ratio to be captured (Default value = 1).

    return_transform : bool

        Whether to apply the transformation to the given data.



    Returns

    -------

    array_like

        If return_transform is True, an array with shape (n_samples, n_components) which is the input samples projected

        onto `n_components` principal components. Otherwise the first `n_components` eigenvectors of the covariance

        matrix corresponding to the `n_components` largest eigenvalues are returned as rows.



    """



    cov_ = np.cov(X, rowvar=False)  # Mean is removed

    evals, evecs = LA.eigh(cov_)  # Get eigenvalues in ascending order, eigenvectors in columns

    evecs = np.fliplr(evecs)  # Flip eigenvectors to get them in descending eigenvalue order



    if var_ratio == 1:

        L = evecs.T

    else:

        evals = np.flip(evals, axis=0)

        var_exp = np.cumsum(evals)

        var_exp = var_exp / var_exp[-1]

        n_components = np.argmax(np.greater_equal(var_exp, var_ratio))

        L = evecs.T[:n_components]  # Set the first n_components eigenvectors as rows of L



    if return_transform:

        return X.dot(L.T)

    else:

        return L





def sum_outer_products(X, weights, remove_zero=False):

    """Computes the sum of weighted outer products using a sparse weights matrix



    Parameters

    ----------

    X : array_like

        An array of data samples with shape (n_samples, n_features_in).

    weights : csr_matrix

        A sparse weights matrix (indicating target neighbors) with shape (n_samples, n_samples).

    remove_zero : bool

        Whether to remove rows and columns of the symmetrized weights matrix that are zero (default: False).



    Returns

    -------

    array_like

        An array with the sum of all weighted outer products with shape (n_features_in, n_features_in).



    """

    weights_sym = weights + weights.T

    if remove_zero:

        _, cols = weights_sym.nonzero()

        ind = np.unique(cols)

        weights_sym = weights_sym.tocsc()[:, ind].tocsr()[ind, :]

        X = X[ind]



    n = weights_sym.shape[0]

    diag = sparse.spdiags(weights_sym.sum(axis=0), 0, n, n)

    laplacian = diag.tocsr() - weights_sym

    sodw = X.T.dot(laplacian.dot(X))



    return sodw





def pairs_distances_batch(X, ind_a, ind_b, batch_size=500):

    """Equivalent to  np.sum(np.square(x[ind_a] - x[ind_b]), axis=1)



    Parameters

    ----------

    X : array_like

        An array of data samples with shape (n_samples, n_features_in).

    ind_a : array_like

        An array of samples indices with shape (m,).

    ind_b : array_like

        Another array of samples indices with shape (m,).

    batch_size :

        Size of each chunk of X to compute distances for (default: 500)



    Returns

    -------

    array-like

        An array of pairwise distances with shape (m,).



    """

    n = len(ind_a)

    res = np.zeros(n)

    for chunk in gen_batches(n, batch_size):

        res[chunk] = np.sum(np.square(X[ind_a[chunk]] - X[ind_b[chunk]]), axis=1)



    return res





def unique_pairs(ind_a, ind_b, n_samples=None):

    """Find the unique pairs contained in zip(ind_a, ind_b)



    Parameters

    ----------

    ind_a : list

        A list with indices of reference samples of length m.

    ind_b : list

        A list with indices of impostor samples of length m.

    n_samples : int, optional

        The total number of samples (= maximum sample index + 1). If None it will be inferred from the indices.



    Returns

    -------

    array-like

         An array of indices of unique pairs with shape (k,) where k <= m.



    """

    # First generate a hash array

    if n_samples is None:

        n_samples = max(np.max(ind_a), np.max(ind_b))



    h = np.array([i * n_samples + j for i, j in zip(ind_a, ind_b)], dtype=np.uint32)



    # Get the indices of the unique elements in the hash array

    _, ind_unique = np.unique(h, return_index=True)



    return ind_unique


def LocalOutlierFactor(dist, n_neighbors, contamination=0.1):
    """
    This is the function used to find outliers using Local outlier factor.
    Parameters:
    dist: array-like, distance matrix with shape (n_predict_samples,n_predict_samples)
    n_neighbors: int,number of nerest neighbors
    contamination: float, the ratio of outlier returned with a range of (0,1), default 0.01
    """

    k_dists = np.partition(dist, n_neighbors - 1, axis=1)

    k_dists = k_dists[:, n_neighbors - 1]

    r_dists = np.maximum(dist, k_dists)

    lrds = n_neighbors / (np.sum(r_dists, axis=1))

    LOF = np.sum(lrds) / (lrds * n_neighbors)

    outliers_index = np.argpartition(LOF, int(dist.shape[0] * contamination))[:int(k_dists.shape[0] * contamination)]

    return outliers_index


def euclidean_distances(X, Y):
    if X.ndim < 2:
        X = X.reshape(1,X.size)
    if Y.ndim < 2:
        Y = Y.reshape(1,Y.size)



    e_dist = X[:, np.newaxis] - Y
    e_dist **= 2
    e_dist = np.sum(e_dist, axis=2)
    e_dist = np.power(e_dist, 0.5)

    return e_dist