from sklearn import tree
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from data_preparation import *

class_names = ['winter', 'spring', 'summer', 'fall']
attributeNames = ['weather', 'temperature', 'humidity', 'wind speed', 'bikes']

y = XD[:, 0]
XD = np.delete(XD, 0, 1)

N, M = XD.shape

# Cross validation outer loop split
K1 = 4
CV1 = model_selection.KFold(K1, shuffle=True)

# Cross validation inner loop split
K2 = 5
CV2 = model_selection.KFold(K2, shuffle=True)

# Parameter values
MAX_SPLITS = 120
min_tree_splits = range(2, MAX_SPLITS + 1)

# errors grabber
Error_test_tree = []
Error_val_tree = np.empty((len(min_tree_splits), K2))

# Parameter values
max_neighbours = 120
neighbours = range(1, max_neighbours + 1)

# errors grabber
Error_test_neighbours = []
Error_val_neighbours = np.empty((len(neighbours), K2))

# Parameter values
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# errors grabber
Error_test_logistic = []
Error_val_logistic = np.empty((len(solvers), K2))

Error_test_baseline = []

# counts = np.bincount(y.astype(int))
# print(np.argmax(counts))

k = 0
# Outer loop
for par_index, test_index in CV1.split(XD, y):
    X_par = XD[par_index, :]
    X_test = XD[test_index, :]
    y_par = y[par_index]
    y_test = y[test_index]

    # print("Outer split no. {0}".format(k+1))
    # Inner loop
    k2 = 0
    for train_index, val_index in CV2.split(X_par, y_par):

        # print("Inner split no. {0}".format(k2+1))
        X_train = X_par[train_index, :]
        X_val = X_par[val_index, :]
        y_train = y_par[train_index]
        y_val = y_par[val_index]

        m_idx = 0
        for s in min_tree_splits:
            dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=s)
            dtc = dtc.fit(X_train, y_train)
            score = dtc.score(X_val, y_val)
            Error_val_tree[m_idx, k2] = 1 - score
            m_idx = m_idx + 1
            # print('errors for minimum tree split {0}: {1}'.format(s, errors))

        m_idx = 0
        for n in neighbours:
            knc = KNeighborsClassifier(n_neighbors=n)
            knc = knc.fit(X_train, y_train)
            score = knc.score(X_val, y_val)
            Error_val_neighbours[m_idx, k2] = 1 - score
            m_idx = m_idx + 1
            # print('errors for minimum tree split {0}: {1}'.format(s, errors))

        m_idx = 0
        for solver in solvers:
            knc = LogisticRegression(solver=solver, max_iter=1000)
            knc = knc.fit(X_train.astype(float), y_train.astype(float))
            score = knc.score(X_val.astype(float), y_val.astype(float))
            Error_val_logistic[m_idx, k2] = 1 - score
            m_idx = m_idx + 1
            # print('errors for minimum tree split {0}: {1}'.format(s, errors))

        k2 = k2 + 1

    # calculate averages and choose model
    averages = np.mean(Error_val_tree, axis=1)
    averages = averages * 100
    idx = np.argmin(averages)
    chosen_split = min_tree_splits[idx]

    # train model on outer split
    dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=chosen_split)
    dtc = dtc.fit(X_par, y_par)
    error = 1 - dtc.score(X_test, y_test)

    # save error
    Error_test_tree.append(error * 100)

    averages = np.mean(Error_val_neighbours, axis=1)
    averages = averages * 100
    idx = np.argmin(averages)
    chosen_n = neighbours[idx]

    # train model on outer split
    dtc = KNeighborsClassifier(n_neighbors=chosen_n)
    dtc = dtc.fit(X_par, y_par)
    error = 1 - dtc.score(X_test, y_test)

    # save error
    Error_test_neighbours.append(error * 100)

    averages = np.mean(Error_val_logistic, axis=1)
    averages = averages * 100
    idx = np.argmin(averages)
    chosen_solver = solvers[idx]

    # train model on outer split
    dtc = LogisticRegression(solver=chosen_solver, max_iter=1000)
    dtc = dtc.fit(X_par.astype(float), y_par.astype(float))
    error = 1 - dtc.score(X_test.astype(float), y_test.astype(float))

    # save error
    Error_test_logistic.append(error * 100)

    score = np.count_nonzero(y_test.astype(int) == 3)
    error = 100 - (score * 100 / y_test.shape[0])
    Error_test_baseline.append(error)

    k = k + 1

# calculate method error
print('\nOverall method error tree: {0} %'.format(np.around(np.mean(Error_test_tree), decimals=2)))
print(np.around(Error_test_tree, decimals=2))

print('\nOverall method error k-neighbours: {0} %'.format(np.around(np.mean(Error_test_neighbours), decimals=2)))
print(np.around(Error_test_neighbours, decimals=2))

print('\nOverall method error logistic regression: {0} %'.format(np.around(np.mean(Error_test_logistic), decimals=2)))
print(np.around(Error_test_logistic, decimals=2))

print('\nOverall method error baseline: {0} %'.format(np.around(np.mean(Error_test_baseline), decimals=2)))
print(np.around(Error_test_baseline, decimals=2))
