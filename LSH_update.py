import timeit

import falconn
import numpy as np
import scipy

"""
This is the implementation of the lsh_maintainer. The input is an instance of LSH index and the original objects.

Functions:
1.__init__:
Parameters:
    lsh: the lsh index
    X: the original data objects

2.update:
Parameters:
    metric_L_old: the old distance metric matrix
    metric_L_new: the new distnace metric matrix
    init_probes: the number of probe searched for out of buckets objects in the second level LSH. Default: 10

Output:
Boolen value of if the update is success. If the new distacne metric is too skewed and cannot be updated, then return False

3.second_up_second_layer
Parameters:
    number_of_tables: The number of hash tables in the second layer of LSH. Default: 10
"""

class lsh_maintainer():
    def __init__(self,lsh,X):
        self.lsh = lsh
        self.X = X
        X_ = X/np.max(np.linalg.norm(X,axis=1))
        X_ = np.append(X_,np.sqrt(1-np.linalg.norm(X_,axis=1)**2)[:,np.newaxis],axis=1)
        self.X_ = X_
        self.cnt_ = 0
        self.index_flag = False

    def update(self, metric_L_new, init_probes = 100):
        if not self.index_flag:
            self.index_flag = True
            self.setup_second_layer()

        # extract the old metric that the hash table is build upon
        metric_L_old = np.copy(self.lsh.metric)

        self.metric_diff = metric_L_new - metric_L_old
        # the old index: a*L_old*x is equal to a*(L_old*L_new^-1)*L_new*x
        value = np.linalg.eigvals(np.dot(metric_L_old, np.linalg.inv(metric_L_new)))
        bound = (scipy.stats.chi2(self.lsh.n_vectors - 1).ppf(0.05), scipy.stats.chi2(self.lsh.n_vectors - 1).ppf(0.95))
        if all(self.lsh.n_vectors * value ** 2 < bound[1]) and all(self.lsh.n_vectors * value ** 2 > bound[0]):
            return

        self.cnt_ += 1

        candidates = []
        for t in range(len(self.lsh.hash_table)):
            table = self.lsh.hash_table[t]
            A_diff = np.dot(table["A"], self.metric_diff)
            for a_idx, a_diff in enumerate(A_diff):
                result = 1e5
                last_round = None
                probes = init_probes
                cur_candidates = []
                while result > self.lsh.width:
                    #check candidates to left boundary and right boundary(L_old-L_new and L_new - L_old)
                    candidate1 = self.query_object.get_unique_candidates(np.append(a_diff,0))
                    candidate2 = self.query_object.get_unique_candidates(np.append(-a_diff,0))
                    candidate = candidate1 + candidate2
                    #sample 100 candidates to estimate the distance to boundary of new sample objects if the set of candidates is too large
                    if len(candidate) > 100:
                        candidate_sample_idx = np.random.choice(np.arange(len(candidate)), 100, replace = False)
                        candidate_sample = np.array(candidate)[candidate_sample_idx]
                    else:
                        candidate_sample = candidate[:]
                    result_ = np.mean(np.abs(np.dot(a_diff, self.X[candidate_sample].T)))
                    if not last_round:
                        result = result_
                    else:
                        result = (result_ * len(candidate) - last_round[0]*len(last_round[1]))/(len(candidate)-len(last_round[1]))
                    last_round = (result_, candidate)
                    probes *= 2
                    self.query_object.set_num_probes(probes)
                    if probes > 10000:
                        print (probes, result, len(candidate))
                        #break the update process if the metric change is too skewed, have to rebuild LSH
                        candidate = range(len(self.X))
                        break

                cur_candidates += candidate

            if len(cur_candidates) > 0:
                #update the hash table
                self.lsh.update_index(list(set(cur_candidates)), metric_L_new, t)
            candidates += cur_candidates

        if len(candidates) > 0:
            self.lsh.set_metric(metric_L_new)

        #check the details of update
        # candidates = list(set(candidates))
        # if len(candidates) > 0:
        #     print("Done...")
        #     print("There are {} objects selected to be rehashed in total".format(len(candidates)))
        #     print("Total LSH update time is {}".format(timeit.default_timer()-start_time))


    def setup_second_layer(self,number_of_tables = 50):
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = self.X.shape[1] + 1
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = number_of_tables
        params_cp.num_rotations = 1
        params_cp.seed = 5721840
        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        falconn.compute_number_of_hash_functions(15, params_cp)

        print('Constructing the LSH table')
        t1 = timeit.default_timer()
        table = falconn.LSHIndex(params_cp)
        self.X_ = self.X_.astype('float')
        table.setup(self.X_)
        t2 = timeit.default_timer()
        print('Done')
        print('Construction time: {}'.format(t2 - t1))

        self.query_object = table.construct_query_object()









