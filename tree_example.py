from sklearn import tree
from sklearn import model_selection
import graphviz

from data_preparation import *

class_names = ['winter', 'spring', 'summer', 'fall']
attributeNames = ['weather', 'temperature', 'humidity', 'wind speed', 'bikes']

y = XD[:, 0]
XD = np.delete(XD, 0, 1)

N, M = XD.shape

# Parameter values
MIN_SPLITS = 43

X_train, X_test, y_train, y_test = model_selection.train_test_split(XD, y)
dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=MIN_SPLITS)
dtc = dtc.fit(X_train, y_train)
score = dtc.score(X_test, y_test)
print(1 - score)

out = tree.export_graphviz(dtc, out_file='tree_gini.gvz', feature_names=attributeNames, class_names=class_names, filled=True)
graphviz.render('dot', 'png', 'tree_gini.gvz', quiet=False)
src = graphviz.Source.from_file('tree_gini.gvz')
## Comment in to automatically open pdf
## Note. If you get an error (e.g. exit status 1), try closing the pdf file/viewer
src.render('../tree_gini', view=True)
