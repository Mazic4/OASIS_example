import numpy as np

"""
This is the implementation of a LSH index that supports step-wise multi-probe hashing.

Functions:
1. __init__: Initialize the LSH index
    Parameters: 
    1.n_vectors: The number of hash vectors in each hash table
    2.width: The width of each bucket
    3.n_probe: The number of probes searched for each query
    4.n_tables: The number of hash tables in the LSH index
    5. random_state: Random seed of numpy

2. build: Build the LSH index
    Parameters:
    1. X: The data to build the index
    2. metric_L: The metric matrix
    
3. query: Given an objects, return the candidate set
    Parameters:
    1. q: The query object
    Output:
    The candidate set
    
4. gen_query_candidate: Generate the query for multi-probe hash of a HASH TABLE
    Parameters:
    A: The set of hash vectors in the given hash table
    B: The set of constant in the given hash table
    w: The width of the hash table
    q: The query object
    n_probe: The number of hash table generated
    
    Output:
    The extra queries that need to be searched
    
5. Update_index: To update the index given the index of to be updated objects in a HASH TABLE
    Parameters:
    1. candidate: the index of to be update objects in a hash table
    2. metric_L: The new metric matrix
    3. t: the index of the hash table
    
"""
class LSH():
    def __init__(self, dataset, n_vectors = 10, width = 12, n_probe=100, n_tables=20, random_state=1):

        self.n_tables = n_tables
        self.n_vectors = n_vectors
        self.width = width
        self.n_probe = n_probe
        np.random.seed(random_state)


    def build(self, X, metric_L = None):
        if metric_L is None:
            metric_L = np.eye(X.shape[1])
        self.X = np.dot(X, metric_L.T)
        self.X_ = X
        self.length = len(X)
        self.hash_table = {}

        self.metric = np.copy(metric_L)

        for table in range(self.n_tables):
            self.hash_table[table] = {}
            self.hash_table[table]["A"] = np.random.multivariate_normal(mean=np.zeros(X.shape[1]),cov=np.eye(X.shape[1]),size=self.n_vectors)
            self.hash_table[table]["B"] = np.random.rand(self.n_vectors,1) * self.width

            self.hash_table[table]["metric"] = metric_L

            self.hash_table[table]["index"] = np.floor((np.dot(self.hash_table[table]["A"],self.X.T) + self.hash_table[table]["B"])/self.width)

            self.hash_table[table]["dict_"] = {}
            self.hash_table[table]["invert_dict"] = {}

            for i in range(self.hash_table[table]["index"].shape[1]):
                #slice the ith column
                idx = ",".join(list(map(str, self.hash_table[table]["index"][:,i].astype(np.int).tolist())))
                if idx not in self.hash_table[table]["dict_"]:
                    self.hash_table[table]["dict_"][idx] = set([i])
                else:
                    self.hash_table[table]["dict_"][idx].add(i)
                self.hash_table[table]["invert_dict"][i] = idx

    def query(self,q_):
        q_ = q_.reshape((1, q_.size))
        metric_L = self.metric
        result = set([])
        for t in self.hash_table:
            table = self.hash_table[t]
            q = np.dot(q_, metric_L.T)
            query_candidates = self.gen_query_candidate(table["A"], table["B"], self.width, q, self.n_probe)
            for index_q_ in query_candidates:
                if index_q_ in self.hash_table[t]["dict_"]:
                    result.update(self.hash_table[t]["dict_"][index_q_])

        return list(result)
    
    def gen_set(self, vec, idx):
        if idx == self.n_vectors:
            return [np.array(vec)]
        vec_ = vec[:]
        vec_[idx] += 1
        return self.gen_set(vec_, idx+1) + self.gen_set(vec, idx+1)
    
    def gen_query_candidate(self, A, B, w, q, n_probe):
        """
        this function is not exactly the same in the Multi-probe paper. However, this function should return
        similar result when the number of vector is not very large.
        """

        #calculate distance to its boundary
        distance_to_boundary_ = (np.dot(A,q.T)+B)/w - np.floor((np.dot(A,q.T)+B)/w)
        #if the distance is larger than 0.5, should consider the distance to the next bucket
        distance_to_boundary = np.where(distance_to_boundary_>0.5, 1-distance_to_boundary_, distance_to_boundary_)
        dict_ = np.where(distance_to_boundary_>0.5, 1, -1)
        #find all possible hash vectors
        idx_candidates = []
        index_q = np.floor((np.dot(A, q.T) + B) / w)
        delta_set = np.array(self.gen_set([0]*self.n_vectors, 0))
        score_set = np.dot(delta_set, distance_to_boundary**2).flatten()
        index_q_candidates = index_q.T + np.multiply(delta_set, dict_[:, 0])[np.argsort(score_set)[:n_probe+1]]
        
        for index_mph in index_q_candidates:
            idx_q = ",".join(list(map(str, index_mph.astype(np.int).tolist())))
            idx_candidates += [idx_q]

        return idx_candidates


    def update_index(self,candidates,metric_L, t):
        X_ = np.dot(self.X_[candidates], metric_L.T)
        self.hash_table[t]["index"][:, candidates] = np.floor(
            (np.dot(self.hash_table[t]["A"], X_.T) + self.hash_table[t]["B"]) / self.width)

        for i in range(len(candidates)):
            idx = ",".join(list(map(str, self.hash_table[t]["index"][:, candidates[i]].astype(np.int).tolist())))
            old_idx = self.hash_table[t]["invert_dict"][candidates[i]]
            self.hash_table[t]["dict_"][old_idx].remove(candidates[i])
            if idx not in self.hash_table[t]["dict_"]:
                self.hash_table[t]["dict_"][idx] = set([candidates[i]])
            else:
                self.hash_table[t]["dict_"][idx].add(candidates[i])
            self.hash_table[t]["invert_dict"][candidates[i]] = idx


    def rebuild(self, X, metric_L = None):
        if metric_L is None:
            metric_L = np.eye(X.shape[1])
        self.X = np.dot(X, metric_L.T)
        self.X_ = X
        self.length = len(X)

        self.metric = metric_L

        for table in range(self.n_tables):

            self.hash_table[table]["index"] = np.floor((np.dot(self.hash_table[table]["A"],self.X.T) + self.hash_table[table]["B"])/self.width)

            self.hash_table[table]["dict_"] = {}
            self.hash_table[table]["invert_dict"] = {}

            for i in range(self.hash_table[table]["index"].shape[1]):
                #slice the ith column
                idx = ",".join(list(map(str, self.hash_table[table]["index"][:,i].astype(np.int).tolist())))
                if idx not in self.hash_table[table]["dict_"]:
                    self.hash_table[table]["dict_"][idx] = set([i])
                else:
                    self.hash_table[table]["dict_"][idx].add(i)
                self.hash_table[table]["invert_dict"][i] = idx

    def set_metric(self, metric):
        self.metric = np.copy(metric)