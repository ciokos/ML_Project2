import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection
import neurolab as nl

from forward_selection import selected_features
from data_preparation import X, attribute_names


y = X[:, -1]
X = X[:, :-1]
N, M = X.shape
attribute_names = attribute_names[:-1]

# Cross validation outer loop split
K1 = 4
CV1 = model_selection.KFold(K1, shuffle=True)

# Cross validation inner loop split
K2 = 5
CV2 = model_selection.KFold(K2, shuffle=True)

# Parameter values
HN = 15
hidden_neurons = range(4, HN)

# ANN parameters
max_epochs = 64  # stop criterion 2 (max epochs in training)
show_error_freq = 5  # frequency of training status updates
learning_goal = 1000000  # stop criterion 1 (train mse to be reached)

# Initialize variables
ANN_Error_test = np.empty((K1, 1))
ANN_HN = np.empty((K1, 1))
LRM_Error_test = np.empty((K1, 1))
BL_Error_test = np.empty((K1, 1))
Error_val = np.empty((len(hidden_neurons), K2))

# Outer loop
k = 0
for par_index, test_index in CV1.split(X, y):
    X_par = X[par_index, :]
    X_test = X[test_index, :]
    y_par = y[par_index]
    y_test = y[test_index]

    print("Outer split no. {0}".format(k+1))
    # Inner loop
    k2 = 0
    for train_index, val_index in CV2.split(X_par, y_par):
        print("Inner split no. {0} in outer split no. {1}".format(k2+1, k+1))
        X_train = X_par[train_index, :]
        X_val = X_par[val_index, :]
        y_train = y_par[train_index]
        y_val = y_par[val_index]

        # Model loop
        m_idx = 0
        for s in hidden_neurons:
            print("Hidden neurons no. {0} in inner split no {1}".format(s, k2+1))
            ann = nl.net.newff([[0, 1]] * M, [s, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
            train_error = ann.train(X_train, y_train.reshape(-1, 1), goal=learning_goal, epochs=max_epochs)
            y_est = ann.sim(X_val).squeeze()
            Error_val[m_idx, k2] = np.sqrt(((np.square(y_val - y_est)).sum())/len(y_val))
            print("RSM: {0}".format(Error_val[m_idx, k2]))
            m_idx += 1

        k2 += 1

    _, S = Error_val.T.shape
    for i in range(S):
        print(np.mean(Error_val.T[:, i]))
    optIdx = np.argmin(np.mean(Error_val.T, axis=0))
    print(optIdx)
    optS = hidden_neurons[optIdx]
    print(optS)
    ANN_HN[k] = optS
    ANN = nl.net.newff([[0, 1]] * M, [optS, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
    ANN.train(X_par, y_par.reshape(-1, 1), goal=learning_goal, epochs=max_epochs)

    ann_y_est = ANN.sim(X_test).squeeze()
    ANN_Error_test[k] = np.sqrt(np.square(y_test - ann_y_est).sum()/len(y_test))

    lrmodel = lm.LinearRegression(fit_intercept=True)
    lrmodel.fit(X_par[:, selected_features], y_par)
    lrm_y_est = lrmodel.predict(X_test[:, selected_features])
    LRM_Error_test[k] = np.sqrt(np.square(y_test - lrm_y_est).sum()/len(y_test))

    bl_y_est = np.ones((len(y_test), 1)) * np.mean(y_par)
    BL_Error_test[k] = np.sqrt(np.square(y_test - bl_y_est).sum()/len(y_test))

    print("ANN Test Error in split {0}: {1}".format(k+1, ANN_Error_test[k]))
    print("LRM Test Error in split {0}: {1}".format(k+1, LRM_Error_test[k]))
    print("BL Test Error in split {0}: {1}".format(k + 1, BL_Error_test[k]))

    k += 1

print("ANN Test Errors:")
print(ANN_Error_test)
# [[1461.37590028]
#  [1423.57682373]
#  [1574.22160315]
#  [1411.47166474]]

print("ANN Hidden Neuron Count:")
print(ANN_HN)
# [[ 4.]
#  [ 9.]
#  [14.]
#  [ 7.]]

print("LRM Test Errors:")
print(LRM_Error_test)
# [[1216.54005002]
#  [1288.61319813]
#  [1297.6187444 ]
#  [1412.03074951]]

print("BL Test Errors:")
print(BL_Error_test)
# [[25456.60116287]
#  [26824.11877553]
#  [27310.62273307]
#  [25497.68938231]]


print("ANN Error: {0}".format(np.mean(ANN_Error_test)))  # 1467.66149797505
print("LRM Generalization Error: {0}".format(np.mean(LRM_Error_test)))  # 1303.7006855130119
print("BL Generalization Error: {0}".format(np.mean(BL_Error_test)))  # 26272.25801344453




