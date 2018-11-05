from sklearn import tree
from sklearn import model_selection
import matplotlib.pyplot as plt

# requires data from exercise 5.1.1
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
MAX_SPLITS = 100
min_tree_splits = range(2, MAX_SPLITS + 1)

# errors grabber
Error_test = []
Error_val = np.empty((len(min_tree_splits), K2))

# for a graph
split_points = []
error_points = []

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
            classes = dtc.predict(X_val)
            error = classes.astype(int) - y_val.astype(int)
            errors = np.count_nonzero(error)
            Error_val[m_idx, k2] = errors
            m_idx = m_idx + 1
            # print('errors for minimum tree split {0}: {1}'.format(s, errors))

        k2 = k2 + 1

    # calculate averages and choose model
    averages = np.mean(Error_val, axis=1)
    idx = np.argmin(averages)
    chosen_split = min_tree_splits[idx]
    print('chosen parameter: {0}'.format(chosen_split))

    # train model on outer split
    dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=chosen_split)
    dtc = dtc.fit(X_par, y_par)
    classes = dtc.predict(X_test)
    error = classes.astype(int) - y_test.astype(int)
    errors = np.count_nonzero(error)
    print('Error: {0} %'.format(errors * 100 / error.shape[0]))

    averages = averages * 100 / error.shape[0]
    plt.plot(min_tree_splits, averages, alpha=0.6, linewidth=2.0)
    plt.plot(chosen_split, np.min(averages), 'kx')

    # save error
    Error_test.append(errors * 100 / error.shape[0])

    k = k + 1

# calculate method error
print('\nOverall method error: {0} %'.format(np.mean(Error_test)))
plt.xlabel('min_samples_split parameter')
plt.ylabel('average error')
plt.show()
