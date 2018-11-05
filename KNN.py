import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

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
max_neighbours = 120
neighbours = range(1, max_neighbours + 1)

# errors grabber
Error_test = []
Error_val = np.empty((len(neighbours), K2))


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
        for n in neighbours:
            knc = KNeighborsClassifier(n_neighbors=n)
            knc = knc.fit(X_train, y_train)
            classes = knc.predict(X_val)
            error = classes.astype(int) - y_val.astype(int)
            errors = np.count_nonzero(error)
            Error_val[m_idx, k2] = errors
            m_idx = m_idx + 1
            # print('errors for minimum tree split {0}: {1}'.format(s, errors))

        k2 = k2 + 1

    # calculate averages and choose model
    averages = np.mean(Error_val, axis=1)
    idx = np.argmin(averages)
    chosen_n = neighbours[idx]
    print('chosen parameter: {0}'.format(chosen_n))

    # train model on outer split
    knc = KNeighborsClassifier(n_neighbors=chosen_n)
    knc = knc.fit(X_par, y_par)
    classes = knc.predict(X_test)
    error = classes.astype(int) - y_test.astype(int)
    errors = np.count_nonzero(error)
    print('Error: {0} %'.format(errors * 100 / error.shape[0]))

    averages = averages * 100 / Error_val.shape[0]
    plt.plot(neighbours, averages, alpha=0.5, linewidth=2.2)
    plt.plot(chosen_n, np.min(averages), 'kx')

    # save error
    Error_test.append(errors * 100 / error.shape[0])

    k = k + 1

# calculate method error
print('\nOverall method error: {0} %'.format(np.around(np.mean(Error_test), decimals=2)))
print(np.around(Error_test, decimals=2))
plt.xlabel('min_samples_split parameter')
plt.ylabel('average error')
plt.show()
