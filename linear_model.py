from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm

from data_preparation import *

# Split dataset into features and target vector
cnt_idx = attribute_names.index('cnt')
y = X[:, cnt_idx]

X_cols = list(range(0, cnt_idx)) + list(range(cnt_idx + 1, len(attribute_names)))
X = X[:, X_cols]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X, y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est - y

# Display scatter plot
figure()
subplot(2, 1, 1)
plot(y, y_est, '.')
xlabel('Bikes count (true)')
ylabel('Bikes count (estimated)')
subplot(2, 1, 2)
hist(residual, 40)

show()
