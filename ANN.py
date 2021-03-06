# exercise 8.2.6

from matplotlib.pyplot import figure, plot, subplot, title, show, bar
import neurolab as nl
from sklearn import model_selection
from scipy import stats
from data_preparation import *

y = X[:, 11]
X = X[:, :-1]
N, M = X.shape

# Normalize data
X = stats.zscore(X);

# Parameters for neural network classifier
n_hidden_units = 8  # number of hidden units
n_train = 2  # number of networks trained in each k-fold
learning_goal = 1000  # stop criterion 1 (train mse to be reached)
max_epochs = 200  # stop criterion 2 (max epochs in training)
show_error_freq = 10  # frequency of training status updates

# K-fold crossvalidation
K = 5  # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

# Variable for classification error
errors = np.zeros(K) * np.nan
error_hist = np.zeros((max_epochs, K)) * np.nan
bestnet = list()
k = 0
for train_index, test_index in CV.split(X, y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))

    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    best_train_error = np.inf
    for i in range(n_train):
        print('Training network {0}/{1}...'.format(i + 1, n_train))
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[0, 1]] * M, [n_hidden_units, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
        if i == 0:
            bestnet.append(ann)
        # train network
        train_error = ann.train(X_train, y_train.reshape(-1, 1), goal=learning_goal, epochs=max_epochs,
                                show=show_error_freq)
        if train_error[-1] < best_train_error:
            bestnet[k] = ann
            best_train_error = train_error[-1]
            error_hist[range(len(train_error)), k] = train_error

    print('Best train error: {0}...'.format(best_train_error))
    y_est = bestnet[k].sim(X_test).squeeze()
    errors[k] = np.power(y_est - y_test, 2).sum().astype(float) / y_test.shape[0]
    k += 1
    # break

# Print the average least squares error
print('Mean-square error: {0}'.format(np.mean(errors)))

figure(figsize=(6, 7));
subplot(2, 1, 1);
bar(range(0, K), errors);
title('Mean-square errors');
subplot(2, 1, 2);
plot(error_hist);
title('Training error as function of BP iterations');
figure(figsize=(6, 7));
subplot(2, 1, 1);
plot(y_est);
plot(y_test);
title('Last CV-fold: est_y vs. test_y');
subplot(2, 1, 2);
plot((y_est - y_test));
title('Last CV-fold: prediction error (est_y-test_y)');
show()

print('Ran Exercise 8.2.6')

# % The weights of the network can be extracted via
bestnet[0].layers[0].np['w']  # Get the weights of the first layer
bestnet[0].layers[0].np['b']  # Get the bias of the first layer
