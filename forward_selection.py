from data_preparation import X, attribute_names
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np

y = X[:,-1]
X = X[:, :-1]
attribute_names = attribute_names[:-1]

textout = ''
internal_cross_validation = 730
# selected_features, features_record, loss_record = feature_selector_lr(X, y, internal_cross_validation, display=textout)

selected_features = [0,  2,  3,  6,  8,  9, 10]

print(attribute_names)
print(selected_features)

fsX = X[:,selected_features]
print(fsX[0])
model = lm.LinearRegression(fit_intercept=True)
model.fit(fsX, y)
print(model.coef_)


# ['spring', '', 'fall', 'winter', '', '', 'rainy', '', 'temp', 'hum', 'windspeed']
# [ 0  2  3  6  8  9 10]


