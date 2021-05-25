import numpy as np

def create_dataset(n_instances, n_testing, path = '../data/',random_state = 265135165):
    ###############load dataset
    # set random seed
    np.random.seed(random_state)

    raw_data = open(path+'all_train.csv').read()

    count = {}
    data = []
    y = []

    for line in raw_data.split('\n')[1:n_instances+5]:
        temp_x = []
        # curline = line.decode("utf-8").split(',')
        curline = line.split(",")
        for i in curline[1:-1]:
            temp_x += [float(i)]

        count[int(temp_x[-1])] = count.get(int(temp_x[-1]), 0) + 1
        data += [temp_x]
        if float(curline[0]) == 1:
            y += [1]
        else:
            y += [2]

        # if int(temp_x[-1]) in count and count[int(temp_x[-1])] >= n_instances/5:
        #     continue
        # else:
        #     count[int(temp_x[-1])] = count.get(int(temp_x[-1]),0) + 1
        #     data += [temp_x]
        #     if float(curline[0]) == 1:
        #         y += [1]
        #     else:
        #         y += [2]

        if len(y) == n_instances:
            break

    data = np.array(data)
    y = np.array(y)

    index = np.arange(len(y))
    np.random.shuffle(index)
    data = data[index]
    y = y[index]

    # y = y[data[:,-1].argsort()]
    # data = data[data[:, -1].argsort()]

    xTraining,yTraining,xTesting,yTesting = data[:n_instances-n_testing],y[:n_instances-n_testing],data[n_instances-n_testing:n_instances],y[n_instances-n_testing:n_instances]

    return xTraining,yTraining,xTesting,yTesting
