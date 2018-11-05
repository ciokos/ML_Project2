from data_preparation import X, attribute_names
from sklearn import preprocessing
import sklearn.linear_model as lm
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show, clim
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np

y = X[:,-1]
X = X[:, :-1]
attribute_names = attribute_names[:-1]

textout = ''
internal_cross_validation = 730
selected_features, features_record, loss_record = feature_selector_lr(X, y, internal_cross_validation, display=textout)

# selected_features = [0,  2,  3,  6,  8,  9, 10]

selected_features = [ 0,  1,  3,  4,  5,  6,  8,  9, 10]

#
# fsX = X[:, selected_features]
# # print(fsX[0])
#
# model = lm.LinearRegression(fit_intercept=True)
# model.fit(fsX, y)
# # model_features = ['spring', 'fall', 'winter', 'rainy',  'temp', 'hum', 'windspeed']
#
#
# # x = fsX[0].reshape(-1,1)
# # print(model.predict(x))
# pred = model.predict(fsX)
#
# attribute_names = np.array(attribute_names)
#
# print(np.sqrt((np.square(y-pred)).sum()/len(y)))
# print(attribute_names[selected_features])
# print(selected_features)
# figure()
# subplot(1,2,1)
# plot(range(1,len(loss_record)), np.sqrt(loss_record[1:]))
# xlabel('Iteration')
# ylabel('Squared error (crossvalidation)')
# subplot(1,3,3)
# bmplot(attribute_names, range(1,features_record.shape[1]), -features_record[:,1:])
# clim(-1.5,0)
# xlabel('Iteration')
# show()

# ['spring', '', 'fall', 'winter', '', '', 'rainy', '', 'temp', 'hum', 'windspeed']
# ['winter', '', 'summer', 'fall', '', '', 'rainy', '', 'temp', 'hum', 'windspeed']
# [ 0  2  3  6  8  9 10]


