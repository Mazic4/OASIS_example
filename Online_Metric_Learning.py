import helpers
import numpy as np


def online_metric_learning(candidate_, new_element_, y_candidate, n_neighbors, L, class_label, lamda):
    """
    This is the online metric learning function.
    Parameters:
    1. candidate_: The set of examples in original space
    2. new_element_: The new incoming object in original space
    3. y_candidate: The label of edach object in the candidate set
    4. n_neighbors: The number of neighbors used in the online metric learining process
    5. L: the old metric matrix
    6. class_label: the label of new_element
    7. lamda: the regularize factor balances the new and old constraints

    Output:
    The new metric matrix
    """
    y_candidate = y_candidate.flatten()
    new_element = np.dot(new_element_, L.T)
    candidate = np.dot(candidate_, L.T)
    same_class_candidate = candidate[y_candidate == class_label]
    diff_class_candidate = candidate[y_candidate != class_label]
    target_dist = np.min(helpers.euclidean_distances(new_element, diff_class_candidate))
    target_neighbors_idx = np.argsort(helpers.euclidean_distances(new_element, same_class_candidate))[0,:n_neighbors]

    for i, knn_neighbor in enumerate(same_class_candidate[target_neighbors_idx]):
        cur_dist = helpers.euclidean_distances(knn_neighbor, new_element)[0]
        if target_dist > cur_dist:
            continue
        #input the objects in the original space
        idx = target_neighbors_idx[i]
        L_new = online_learning(candidate_[y_candidate == class_label][idx], new_element_, target_dist**2, L, lamda)
        L = np.copy(L_new)


    target_dist = np.sort(helpers.euclidean_distances(new_element, same_class_candidate))[0,n_neighbors]
    target_neighbors_idx = np.argsort(helpers.euclidean_distances(new_element, diff_class_candidate))[0,:n_neighbors]

    for i, knn_neighbor in enumerate(diff_class_candidate[target_neighbors_idx]):
        cur_dist = helpers.euclidean_distances(knn_neighbor, new_element)[0]
        if target_dist < cur_dist:
            continue
        idx = target_neighbors_idx[i]
        L_new = online_learning(candidate_[y_candidate != class_label][idx], new_element_, target_dist**2, L, lamda)
        L = np.copy(L_new)

    return L

def online_learning(x1, x2, target_distance, L, lamda):
    """
    This function implement the online metric update step.
    Parameters:
        1. x1, x2: The pair of two objects
        2. target_distance: Target distance between the two objects
        3. L: The old metric matrix
        4. lamda: the regularize factor balances the new and old constraints

    Output:
        new metric matrix
    """
    z = (x1 - x2).T

    A = np.dot(L.T, L)
    yt_hat = np.dot(np.dot(z.T, A), z)

    y_new = (lamda * target_distance * yt_hat - 1 + ((lamda * target_distance * yt_hat - 1) ** 2 + 4 * lamda * yt_hat ** 2) ** 0.5) / (2 * lamda * yt_hat)

    beta = -lamda * (y_new - target_distance) / (1 + lamda * (y_new - target_distance) * yt_hat)

    alpha = ((1 + beta * np.dot(np.dot(z.T, A), z)) ** 0.5 - 1) / np.dot(np.dot(z.T, A), z)

    L_new = L + alpha * np.dot(np.dot(L, z), np.dot(z.T, A))

    return L_new
