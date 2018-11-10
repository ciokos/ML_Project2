from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
from scipy import stats
import numpy as np

K = 4
Error_test_tree = np.array([34.97, 38.8, 38.25, 36.81]).reshape(K, 1)
Error_test_logistic = np.array([39.34, 37.7, 39.89, 38.46]).reshape(K, 1)
Error_test_baseline = np.array([71.58, 74.86, 73.77, 76.92]).reshape(K, 1)

# tree vs logistic

print('tree vs logistic')

z = (Error_test_tree - Error_test_logistic)
zb = z.mean()
nu = K - 1
sig = (z - zb).std() / np.sqrt(K - 1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha / 2, nu)
zH = zb + sig * stats.t.ppf(1 - alpha / 2, nu)

print([zL, zH])

if zL <= 0 and zH >= 0:
    print('Classifiers are not significantly different')
else:
    print('Classifiers are significantly different.')

# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_test_tree, Error_test_logistic), axis=1))
xlabel('Decision Tree   vs.   Logistic Regression')
ylabel('Cross-validation error [%]')

show()

# tree vs baseline

print('tree vs baseline')

z = (Error_test_tree - Error_test_baseline)
zb = z.mean()
nu = K - 1
sig = (z - zb).std() / np.sqrt(K - 1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha / 2, nu)
zH = zb + sig * stats.t.ppf(1 - alpha / 2, nu)

print([zL, zH])

if zL <= 0 and zH >= 0:
    print('Classifiers are not significantly different')
else:
    print('Classifiers are significantly different.')

# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_test_tree, Error_test_baseline), axis=1))
xlabel('Decision Tree   vs.   Baseline')
ylabel('Cross-validation error [%]')

show()

# logistic vs baseline

print('logistic vs baseline')

z = (Error_test_logistic - Error_test_baseline)
zb = z.mean()
nu = K - 1
sig = (z - zb).std() / np.sqrt(K - 1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha / 2, nu)
zH = zb + sig * stats.t.ppf(1 - alpha / 2, nu)

print([zL, zH])

if zL <= 0 and zH >= 0:
    print('Classifiers are not significantly different')
else:
    print('Classifiers are significantly different.')

# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_test_logistic, Error_test_baseline), axis=1))
xlabel('Logistic Regression   vs.   Baseline')
ylabel('Cross-validation error [%]')

show()
