"""

Large Margin Nearest Neighbor Classification

"""



# Author: John Chiotellis <johnyc.code@gmail.com>



# License: BSD 3 clause (C) John Chiotellis



from __future__ import print_function

import logging
import os
import sys

import numpy as np
from helpers import unique_pairs, pairs_distances_batch, sum_outer_products, pca_fit
from scipy import sparse, optimize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import gen_batches
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y, check_random_state


class LargeMarginNearestNeighbor(KNeighborsClassifier):

    """Large Margin Nearest Neighbor metric learning.

    

    This implementation follows closely Kilian Weinberger's MATLAB code found at

    https://bitbucket.org/mlcircus/lmnn which solves the unconstrained problem, finding a linear

    transformation with L-BFGS instead of solving the constrained problem that finds the globally

    optimal metric.



    Parameters

    ----------

    L : array_like

        Initial transformation in an array with shape (n_features_out, n_features_in).  If None `load`

        will be used to load a transformation from a file. (default: None)



    n_neighbors : int

        Number of target neighbors (default: 3)



    max_iter : int

        Maximum number of iterations in the optimization (default: 200)



    use_pca : bool

        Whether to use pca to warm-start the linear transformation.

        If False, the identity will be used. (default: True)



    tol : float

        Tolerance for the optimization  (default: 1e-5)



    n_features_out : int

        Preferred dimensionality of the inputs after the transformation.

        If None it is inferred from `use_pca` and `L`.(default: None)



    max_constr : int

        Maximum number of constraints to enforce per iteration (default: 10 million).



    use_sparse : bool

        Whether to use a sparse or a dense matrix for the impostor-pairs storage. Using a sparse matrix,

        the distance to impostors is computed twice, but it is somewhat faster for

        larger data sets than using a dense matrix. With a dense matrix, the unique impostor pairs have to be identified

        explicitly (default: True).



    load : string

        A file path from which to load a linear transformation.

        If None, either identity or pca will be used based on `use_pca` (default: None).



    save : string

        A file path prefix to save intermediate linear transformations to. After every function

        call, it will be extended with the function call number and the `.npy` file

        extension. If None, nothing will be saved (default: None).



    verbose : int

        The level of logger verbosity. Can take values from 0 to 4 inclusive (default: 0).

        0: Only basic information will be printed.

        1: Information from the classifier will be logged.

        2: Information from the classifier and debugging information will be logged.

        3: Information from the classifier and the L-BFGS optimizer will be logged.

        4: Information from the classifier, the L-BFGS optimizer and debugging information will be logged.



    random_state : int

        A seed for reproducibility of random state  (default: None).



    Attributes

    ----------

    L_ : array_like

        The linear transformation used during fitting with shape (n_features_out, n_features_in).



    n_neighbors_ : int

        The number of target neighbors (decreased if n_neighbors was not realistic for all classes).



    n_features_out_ : int

        The dimensionality of a vector after applying to it the linear transformation.



    X_ : array_like

        An array of training samples with shape (n_samples, n_features_in).



    y_ : array_like

        An array of training labels with shape (n_samples,).



    labels_: array_like

        An array of the uniquely appearing class labels with shape (n_classes,) and type object.



    classes_: array_like

        An array of the uniquely appearing class labels as integers with shape (n_classes,) and type int.



    targets_ : array_like

        An array of target neighbors for each sample with shape (n_samples, n_neighbors).



    grad_static_ : array_like

        An array of the gradient component caused by target neighbors, that stays fixed throughout the algorithm with

        shape (n_features_in, n_features_in).



    n_iter_ : int

        The number of iterations of the optimizer.



    n_funcalls_ : int

        The number of times the optimizer computes the loss and the gradient.



    name : str

        A name for the instance based on the current number of existing instances.



    logger : object

        A logger object to log information during fitting.



    details_ : dict

        A dictionary of information created by the L-BFGS optimizer during fitting.



    _obj_count : int (class attribute)

        An instance counter



    Examples

    --------

    >>> X = [[0], [1], [2], [3]]

    >>> y = [0, 0, 1, 1]

    >>> from pylmnn.lmnn import LargeMarginNearestNeighbor

    >>> lmnn = LargeMarginNearestNeighbor(n_neighbors=1)

    >>> lmnn.fit(X, y) # doctest: +ELLIPSIS

    LargeMarginNearestNeighbor(L=None, load=None, max_constr=10000000,

              max_iter=200, n_features_out=None, n_neighbors=1,

              random_state=None, save=None, tol=1e-05, use_pca=True,

              use_sparse=True, verbose=1)

    >>> print(lmnn.predict([[1.1]]))

    [0]

    >>> print(lmnn.predict_proba([[0.9]]))

    [[ 1.  0.]]



    """



    _obj_count = 0



    def __init__(self, L=None, n_neighbors=3, n_features_out=None, max_iter=200, tol=1e-5, use_pca=True,

                 max_constr=int(1e7), use_sparse=True, load=None, save=None, verbose=0, random_state=None):



        super(LargeMarginNearestNeighbor, self).__init__(n_neighbors=n_neighbors)



        # Parameters

        self.L = L

        self.n_features_out = n_features_out

        self.max_iter = max_iter

        self.tol = tol

        self.use_pca = use_pca

        self.max_constr = max_constr

        self.use_sparse = use_sparse

        self.load = load

        self.save = save

        self.verbose = verbose

        self.random_state = random_state



        # Setup instance name and logger

        LargeMarginNearestNeighbor._obj_count += 1

        self.name = __name__ + '(' + str(LargeMarginNearestNeighbor._obj_count) + ')'

        self.logger = self._setup_logger()



    def fit(self, X, y):

        """Find a linear transformation by optimization of the unconstrained problem, such that the k-nearest neighbor

        classification accuracy improves.



        Parameters

        ----------

        X : array_like

            An array of training samples with shape (n_samples, n_features_in).

        y : array_like

            An array of data labels with shape (n_samples,).



        Returns

        -------

        LargeMarginNearestNeighbor

            self



        """



        # Check inputs consistency

        self.X_, y = check_X_y(X, y)

        check_classification_targets(y)



        # Store the appearing classes and the class index for each sample

        self.labels_, self.y_ = np.unique(y, return_inverse=True)

        self.classes_ = np.arange(len(self.labels_))



        # Check that the number of neighbors is achievable for all classes

        self.n_neighbors_ = self.check_n_neighbors(self.y_)

        # TODO: Notify superclass KNeighborsClassifier that n_neighbors might have changed to n_neighbors_

        # super().set_params(n_neighbors=self.n_neighbors_)



        # Initialize transformer

        self.L_, self.n_features_out_ = self._init_transformer()



        # Prepare for saving if needed

        if self.save is not None:

            save_dir, save_file = os.path.split(self.save)

            if save_dir != '' and not os.path.exists(save_dir):

                os.mkdir(save_dir)

            save_file = self.save + '_' + str(self.n_funcalls_)

            np.save(save_file, self.L_)



        # Find target neighbors (fixed)

        self.targets_ = self._select_target_neighbors()



        # Compute gradient component of target neighbors (constant)

        self.grad_static_ = self._compute_grad_static()



        # Initialize number of optimizer iterations and objective function calls

        self.n_iter_ = 0

        self.n_funcalls_ = 0



        # Call optimizer

        disp = 1 if self.verbose in [3, 4] else None

        self.logger.info('Now optimizing...')

        L, loss, details = optimize.fmin_l_bfgs_b(func=self._loss_grad, x0=self.L_, bounds=None,

                                                  m=100, pgtol=self.tol, maxfun=500*self.max_iter,

                                                  maxiter=self.max_iter, disp=disp, callback=self._cb)

        # Reshape result from optimizer

        self.L_ = L.reshape(self.n_features_out_, L.size // self.n_features_out_)



        # Store output to return

        self.details_ = details

        self.details_['loss'] = loss



        # Fit a simple nearest neighbor classifier with the learned metric

        super(LargeMarginNearestNeighbor, self).fit(self.transform(), y)



        return self



    def transform(self, X=None):

        """Applies the learned transformation to the inputs.



        Parameters

        ----------

        X : array_like

            An array of data samples with shape (n_samples, n_features_in) (default: None, defined when fit is called).



        Returns

        -------

        array_like

            An array of transformed data samples with shape (n_samples, n_features_out).



        """

        if X is None:

            X = self.X_

        else:

            X = check_array(X)



        return X.dot(self.L_.T)



    def predict(self, X):

        """Predict the class labels for the provided data



        Parameters

        ----------

        X : array-like, shape (n_query, n_features)

            Test samples.



        Returns

        -------

        y_pred : array of shape [n_query]

            Class labels for each data sample.

        """



        # Check if fit had been called

        check_is_fitted(self, ['X_', 'y_'])

        y_pred = super(LargeMarginNearestNeighbor, self).predict(self.transform(X))



        return y_pred



    def predict_proba(self, X):

        """Return probability estimates for the test data X.



        Parameters

        ----------

        X : array-like, shape (n_query, n_features)

            Test samples.



        Returns

        -------

        p : array of shape = [n_samples, n_classes], or a list of n_outputs

            of such arrays if n_outputs > 1.

            The class probabilities of the input samples. Classes are ordered

            by lexicographic order.

        """



        # Check if fit had been called

        check_is_fitted(self, ['X_', 'y_'])

        probabilities = super(LargeMarginNearestNeighbor, self).predict_proba(self.transform(X))



        return probabilities



    def _setup_logger(self):

        """Instantiate a logger object for the current class instance"""

        logger = logging.getLogger(self.name)



        if self.verbose in [1, 3]:

            logger.setLevel(logging.INFO)

        elif self.verbose in [2, 4]:

            logger.setLevel(logging.DEBUG)

        else:

            logger.setLevel(logging.NOTSET)



        stream_handler = logging.StreamHandler(stream=sys.stdout)

        formatter = logging.Formatter(fmt='%(asctime)s  %(name)s - %(levelname)s : %(message)s')

        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)



        return logger



    def check_n_neighbors(self, y, n_neighbors=None):

        """Check if all classes have enough samples to query the specified number of neighbors."""



        if n_neighbors is None:

            n_neighbors = self.n_neighbors



        min_class_size = np.bincount(y).min()

        if min_class_size < 2:

            raise ValueError('At least one class has less than 2 ({}) training samples.'.format(min_class_size))



        max_neighbors = min_class_size - 1

        if n_neighbors > max_neighbors:

            self.logger.warning('n_neighbors(={}) too high. Setting to {}\n'.format(n_neighbors, max_neighbors))



        return min(n_neighbors, max_neighbors)



    def _init_transformer(self):

        """Initialize the linear transformation by setting to user specified parameter, loading from a file,

        applying PCA or setting to identity."""



        if self.L is not None:

            L = self.L

        elif self.load is not None:

            L = np.load(self.load)

        elif self.use_pca and self.X_.shape[1] > 1:

            L = pca_fit(self.X_, return_transform=False)

        else:

            L = np.eye(self.X_.shape[1])



        n_features_out = L.shape[0] if self.n_features_out is None else self.n_features_out

        n_features_in = self.X_.shape[1]



        if L.shape[1] != n_features_in:

            raise ValueError('Dimensionality of the given transformation and the inputs don\'t match ({},{}).'.format(L.shape[1], n_features_in))



        if n_features_out > n_features_in:

            self.logger.warning('n_features_out({}) cannot be larger than the inputs dimensionality, setting n_features_out to {}!'.format(n_features_out, n_features_in))

            n_features_out = n_features_in



        if L.shape[0] > n_features_out:

            L = L[:n_features_out]



        return L, n_features_out



    def _select_target_neighbors(self):

        """Find the target neighbors of each sample, that stay fixed during training.



        Returns

        -------

        array_like

            An array of neighbors indices for each sample with shape (n_samples, n_neighbors).



        """



        self.logger.info('Finding target neighbors...')

        target_neighbors = np.empty((self.X_.shape[0], self.n_neighbors_), dtype=int)

        for class_ in self.classes_:

            class_ind, = np.where(np.equal(self.y_, class_))

            dist = euclidean_distances(self.X_[class_ind], squared=True)

            np.fill_diagonal(dist, np.inf)

            neigh_ind = np.argpartition(dist, self.n_neighbors_ - 1, axis=1)

            neigh_ind = neigh_ind[:, :self.n_neighbors_]

            # argpartition doesn't guarantee sorted order, so we sort again but only the k neighbors

            row_ind = np.arange(len(class_ind))[:, None]

            neigh_ind = neigh_ind[row_ind, np.argsort(dist[row_ind, neigh_ind])]

            target_neighbors[class_ind] = class_ind[neigh_ind]



        return target_neighbors



    def _compute_grad_static(self):

        """Compute the gradient component due to the target neighbors that stays fixed throughout training



        Returns

        -------

        array_like

            An array with the sum of all weighted outer products with shape (n_features_in, n_features_in).



        """



        self.logger.info('Computing gradient component due to target neighbors...')

        n_samples, n_neighbors = self.targets_.shape

        rows = np.repeat(np.arange(n_samples), n_neighbors)  # 0 0 0 1 1 1 ... (n-1) (n-1) (n-1) with n_neighbors=3

        cols = self.targets_.flatten()

        targets_sparse = sparse.csr_matrix((np.ones(n_samples * n_neighbors), (rows, cols)), shape=(n_samples, n_samples))



        return sum_outer_products(self.X_, targets_sparse)



    def _cb(self, L):

        """Callback function called after every iteration of the optimizer. The intermediate transformations are

        saved to files if a valid `save` parameter was passed.



        Parameters

        ----------

        L : array_like

            The (flattened) linear transformation in the current iteration.



        """

        self.logger.info('Iteration {:4} / {:4}'.format(self.n_iter_, self.max_iter))

        if self.save is not None:

            save_file = self.save + '_' + str(self.n_iter_)

            L = L.reshape(self.n_features_out_, L.size // self.n_features_out_)

            np.save(save_file, L)

        self.n_iter_ += 1



    def _loss_grad(self, L):

        """Compute the loss under a given linear transformation `L` and the loss gradient w.r.t. `L`.



        Parameters

        ----------

        L : array_like

            The current (flattened) linear transformation with shape (n_features_out x n_features_in,).



        Returns

        -------

        tuple

            float: The new loss.

            array_like: The new (flattened) gradient with shape (n_features_out x n_features_in,).



        """



        n_samples, n_features_in = self.X_.shape

        self.L_ = L.reshape(self.n_features_out_, n_features_in)

        self.n_funcalls_ += 1

        self.logger.debug('Function call {}'.format(self.n_funcalls_))



        Lx = self.transform()



        # Compute distances to target neighbors under L (plus margin)

        self.logger.debug('Computing distances to target neighbors under new L...')

        dist_tn = np.zeros((n_samples, self.n_neighbors_))

        for k in range(self.n_neighbors_):

            dist_tn[:, k] = np.sum(np.square(Lx - Lx[self.targets_[:, k]]), axis=1) + 1



        # Compute distances to impostors under L

        self.logger.debug('Setting margin radii...')

        margin_radii = np.add(dist_tn[:, -1], 2)



        imp1, imp2, dist_imp = self._find_impostors(Lx, margin_radii, use_sparse=self.use_sparse)



        self.logger.debug('Computing loss and gradient under new L...')

        loss = 0

        A0 = sparse.csr_matrix((n_samples, n_samples))

        for k in reversed(range(self.n_neighbors_)):

            loss1 = np.maximum(dist_tn[imp1, k] - dist_imp, 0)

            act, = np.where(loss1 != 0)

            A1 = sparse.csr_matrix((2*loss1[act], (imp1[act], imp2[act])), (n_samples, n_samples))



            loss2 = np.maximum(dist_tn[imp2, k] - dist_imp, 0)

            act, = np.where(loss2 != 0)

            A2 = sparse.csr_matrix((2*loss2[act], (imp1[act], imp2[act])), (n_samples, n_samples))



            vals = np.squeeze(np.asarray(A2.sum(0) + A1.sum(1).T))

            A0 = A0 - A1 - A2 + sparse.csr_matrix((vals, (range(n_samples), self.targets_[:, k])), (n_samples, n_samples))

            loss = loss + np.sum(loss1 ** 2) + np.sum(loss2 ** 2)



        grad_new = sum_outer_products(self.X_, A0, remove_zero=True)

        df = self.L_.dot(self.grad_static_ + grad_new)

        df *= 2

        loss = loss + (self.grad_static_ * (self.L_.T.dot(self.L_))).sum()

        self.logger.info('Loss = {:,} at function call {}.\n'.format(loss, self.n_funcalls_))



        return loss, df.flatten()



    def _find_impostors(self, Lx, margin_radii, use_sparse=True):

        """Compute all impostor pairs exactly.



        Parameters

        ----------

        Lx : array_like

            An array of transformed samples with shape (n_samples, n_features_out).

        margin_radii : array_like

            An array of distances to the farthest target neighbors + margin, with shape (n_samples,).

        use_sparse : bool

            Whether to use a sparse matrix for storing the impostor pairs (default: True).



        Returns

        -------

        tuple: (array_like, array_like, array_like)



            imp1 : array_like

                An array of sample indices with shape (n_impostors,).

            imp2 : array_like

                An array of sample indices that violate a margin with shape (n_impostors,).

            dist : array_like

                An array of pairwise distances of (imp1, imp2) with shape (n_impostors,).



        """

        n_samples = Lx.shape[0]



        self.logger.debug('Now computing impostor vectors...')

        if use_sparse:

            # Initialize impostors matrix

            impostors_sp = sparse.csr_matrix((n_samples, n_samples), dtype=np.int8)



            for class_ in self.classes_[:-1]:

                imp1, imp2 = [], []

                ind_in, = np.where(np.equal(self.y_, class_))

                ind_out, = np.where(np.greater(self.y_, class_))



                # Subdivide idx_out x idx_in to chunks of a size that is fitting in memory

                self.logger.debug('Impostor classes {} to class {}..'.format(self.classes_[self.classes_ > class_], class_))

                ii, jj = self._find_impostors_batch(Lx[ind_out], Lx[ind_in], margin_radii[ind_out],

                                                    margin_radii[ind_in])

                if len(ii):

                    imp1.extend(ind_out[ii])

                    imp2.extend(ind_in[jj])

                    new_imps = sparse.csr_matrix(([1] * len(imp1), (imp1, imp2)), shape=(n_samples, n_samples),

                                                 dtype=np.int8)

                    impostors_sp = impostors_sp + new_imps



            imp1, imp2 = impostors_sp.nonzero()

            # subsample constraints if they are too many

            if impostors_sp.nnz > self.max_constr:

                random_state = check_random_state(self.random_state)

                ind_subsample = random_state.choice(impostors_sp.nnz, self.max_constr, replace=False)

                imp1, imp2 = imp1[ind_subsample], imp2[ind_subsample]



            # self.logger.debug('Computing distances to impostors under new L...')

            dist = pairs_distances_batch(Lx, imp1, imp2)

        else:

            # Initialize impostors vectors

            imp1, imp2, dist = [], [], []

            for class_ in self.classes_[:-1]:

                ind_in, = np.where(np.equal(self.y_, class_))

                ind_out, = np.where(np.greater(self.y_, class_))

                # Permute the indices (experimental)

                # idx_in = np.random.permutation(idx_in)

                # idx_out = np.random.permutation(idx_out)



                # Subdivide idx_out x idx_in to chunks of a size that is fitting in memory

                self.logger.debug('Impostor classes {} to class {}..'.format(self.classes_[self.classes_ > class_], class_))

                ii, jj, dd = self._find_impostors_batch(Lx[ind_out], Lx[ind_in], margin_radii[ind_out],

                                                        margin_radii[ind_in], return_dist=True)

                if len(ii):

                    imp1.extend(ind_out[ii])

                    imp2.extend(ind_in[jj])

                    dist.extend(dd)



            ind_unique = unique_pairs(imp1, imp2, n_samples)

            self.logger.debug('Found {} unique pairs out of {}.'.format(len(ind_unique), len(imp1)))



            # subsample constraints if they are too many

            if len(ind_unique) > self.max_constr:

                random_state = check_random_state(self.random_state)

                ind_unique = random_state.choice(ind_unique, self.max_constr, replace=False)



            imp1 = np.asarray(imp1)[ind_unique]

            imp2 = np.asarray(imp2)[ind_unique]

            dist = np.asarray(dist)[ind_unique]



        return imp1, imp2, dist



    @staticmethod

    def _find_impostors_batch(x1, x2, t1, t2, return_dist=False, batch_size=500):

        """Find impostor pairs in chunks to avoid large memory usage



        Parameters

        ----------

        x1 : array_like

            An array of transformed data samples with shape (n_samples, n_features).

        x2 : array_like

            An array of transformed data samples with shape (m_samples, n_features) where m_samples < n_samples.

        t1 : array_like

            An array of distances to the margins with shape (n_samples,).

        t2 : array_like

            An array of distances to the margins with shape (m_samples,).

        batch_size : int (Default value = 500)

            The size of each chunk of x1 to compute distances to.

        return_dist : bool (Default value = False)

            Whether to return the distances to the impostors.



        Returns

        -------

        tuple: (array_like, array_like, [array_like])

            

            imp1 : array_like

                An array of sample indices with shape (n_impostors,).

            imp2 : array_like

                An array of sample indices that violate a margin with shape (n_impostors,).

            dist : array_like, optional

                An array of pairwise distances of (imp1, imp2) with shape (n_impostors,).



        """



        n, m = len(t1), len(t2)

        imp1, imp2, dist = [], [], []

        for chunk in gen_batches(n, batch_size):

            dist_out_in = euclidean_distances(x1[chunk], x2, squared=True)

            i1, j1 = np.where(dist_out_in < t1[chunk, None])

            i2, j2 = np.where(dist_out_in < t2[None, :])

            if len(i1):

                imp1.extend(i1 + chunk.start)

                imp2.extend(j1)

                if return_dist:

                    dist.extend(dist_out_in[i1, j1])

            if len(i2):

                imp1.extend(i2 + chunk.start)

                imp2.extend(j2)

                if return_dist:

                    dist.extend(dist_out_in[i2, j2])



        if return_dist:

            return imp1, imp2, dist

        else:

            return imp1, imp2



    def __getstate__(self):

        """Have to override getstate because logger is not picklable"""

        state = dict(self.__dict__)

        del state['logger']



        return state



    def __setstate__(self, state):

        """Have to override setstate because logger is not picklable"""

        self.__dict__.update(state)

        self.logger = self._setup_logger()