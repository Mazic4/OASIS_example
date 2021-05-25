import timeit

import LSH
import LSH_update
import Online_Metric_Learning
import create_dataset_made_
import create_dataset_hepmass
import helpers
import numpy as np
from scipy.stats import mode

def OASIS_example(dataset, update_metric, update_lsh):
    print ("Dataset:", dataset)
    print ("Metric update method:", update_metric)
    print ("LSH update method:", update_lsh)
    total_lsh_update_time = 0
    # update_metric = 'online'
    # update_lsh = 'rebuild'
    X_example,y_example = X[:100],y[:100]
    if dataset == "hepmass":
        tn_neighbors = 10
        lamda = 0.001
    elif dataset == "made":
        tn_neighbors = 10
        lamda = 0.001

    lsh_maintainer_flag = False

    recalls_10 = []
    recalls_10_ = []
    recalls_50 = []
    recalls_50_ = []
    recalls_100 = []
    recalls_100_ = []
    result_acc = 0
    accuracy = 0
    metric_L = np.eye(X_test.shape[1])

    #build the LSH index
    t0 = timeit.default_timer()
    index_flag = True
    metric_build = np.copy(metric_L)
    # parameters for synthetic dataset
    if dataset == "made":
        width = 12
    # parameters for hepmass
    elif dataset == "hepmass":
        width = 7
    lsh = LSH.LSH(dataset, n_vectors=10, width = width)
    lsh.build(X,metric_build)
    total_lsh_build_time = timeit.default_timer() - t0
    print ("LSH Index Build...")
    print ("Total LSH index building time is", total_lsh_build_time)

    lsh_ = LSH.LSH(dataset, n_vectors=10, width=width)
    lsh_.build(X, metric_build)

    for i in range(X_test.shape[0]):
        print (i)

        new_element = np.dot(X_test[i], metric_L.T)

        #extract candidates
        candidate = lsh.query(X_test[i])
        candidate_ = lsh_.query(X_test[i])
        #extract ground truth distance
        training_set_ = np.dot(X,metric_L.T)

        #extract the ground truth n_neighbors
        e2distances_ = helpers.euclidean_distances(new_element, training_set_)

        #calculate the recall varying number of n
        n_neighbors = 10
        knn_points_index_ = np.argsort(e2distances_)[0, :n_neighbors]
        recalls_10 += [len(np.intersect1d(candidate,knn_points_index_))]
        recalls_10_ += [len(np.intersect1d(candidate_, knn_points_index_))]
        result_acc += np.sum(y[knn_points_index_] == y_test[i])

        n_neighbors = 50
        knn_points_index_ = np.argsort(e2distances_)[0, :n_neighbors]
        recalls_50 += [len(np.intersect1d(candidate, knn_points_index_))]
        recalls_50_ += [len(np.intersect1d(candidate_, knn_points_index_))]

        n_neighbors = 100
        knn_points_index_ = np.argsort(e2distances_)[0, :n_neighbors]
        recalls_100 += [len(np.intersect1d(candidate, knn_points_index_))]
        recalls_100_ += [len(np.intersect1d(candidate_, knn_points_index_))]

        #perform KNN classification on the example set
        class_label = y_test[i]
        example_set = np.dot(X_example, metric_L.T)
        knn_pred_idx = np.argsort(helpers.euclidean_distances(new_element, example_set))[0, :n_neighbors]
        knn_pred = mode(y_example[knn_pred_idx])[0][0]
        cor_pred = (knn_pred == class_label)
        accuracy += cor_pred

        print (len(candidate), len(candidate_))
        print(recalls_10[-1], recalls_10_[-1])
        print(recalls_50[-1], recalls_50_[-1])
        print(recalls_100[-1], recalls_100_[-1])
        # print (np.mean(recalls_10), np.mean(recalls_10_))
        # print (np.mean(recalls_50), np.mean(recalls_50_))
        # print (np.mean(recalls_100), np.mean(recalls_100_))

        if update_metric == 'online':
            #insert the example if KNN classification is incorrect
            if not cor_pred:
                X_example = np.append(X_example, X_test[i][np.newaxis,], axis=0)
                y_example = np.append(y_example, y_test[i])
            #update the metric on new object
            metric_L = Online_Metric_Learning.online_metric_learning \
            (X_example, X_test[i][np.newaxis,], y_example,tn_neighbors, metric_L, class_label, lamda)

        t0 = timeit.default_timer()
        if update_lsh == 'rebuild':
            lsh.build(X, metric_L)

        elif update_lsh == 'online':
            #if the first time perform online update, then initialize the LSH maintainer
            if not lsh_maintainer_flag:
                lsh_maintainer = LSH_update.lsh_maintainer(lsh, X)
                lsh_maintainer_flag = True

            lsh_maintainer.update(metric_L)
            lsh_.build(X, metric_L)

        total_lsh_update_time += timeit.default_timer() - t0

    print ("The accuracy of KNN Classification is:", accuracy/len(y_test))
    print ("The accuracy of KNN query when n = 10 is:", result_acc/10/len(y_test))

    print ("Tototal lsh update time is:", total_lsh_update_time)
    print ("The recall when n = 10:", sum(recalls_10)/10/len(y_test))
    print ("The recall when n = 50:", sum(recalls_50)/50/len(y_test))
    print ("The recall when n = 100:", sum(recalls_100)/100/len(y_test))

if __name__ == "__main__":

    dataset = "hepmass"  # made or hepmass

    if dataset == "made":
        X, y, X_test, y_test = create_dataset_made_.create_dataset(n_instances=10000, n_testing=100)
    elif dataset == "hepmass":
        X, y, X_test, y_test = create_dataset_hepmass.create_dataset(n_instances = 10000, n_testing= 100, random_state=200)

    print ("The number of testing queries is:", X_test.shape[0])

    OASIS_example(dataset, update_metric="online", update_lsh="online")