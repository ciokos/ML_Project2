from scipy import stats
from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show, plot, xticks
import numpy as np


K = 4

ANN_Error = np.array([1461.37590028, 1423.57682373, 1574.22160315, 1411.47166474]).reshape(K, 1)
LRM_Error = np.array([1216.54005002, 1288.61319813, 1297.6187444, 1412.03074951]).reshape(K, 1)
BL_Error = np.array([25456.60116287, 26824.11877553, 27310.62273307, 25497.68938231]).reshape(K ,1)

HN = [4, 9, 14, 7]
z = np.abs(BL_Error - ANN_Error)
zb = z.mean()
nu = K-1
sig = (z-zb).std() / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu)
zH = zb + sig * stats.t.ppf(1-alpha/2, nu)
print([zL, zH])
# figure()
# plot(HN, ANN_Error, 'o')
# xlabel("Hidden neurons")
# ylabel("Error")
# xticks(np.arange(np.min(HN), np.max(HN)+1, 1))
# show()
#
z_ann_lrm = np.abs(ANN_Error - LRM_Error)
zb_ann_lrm = z_ann_lrm.mean()
nu = K-1
sig = (z_ann_lrm-zb_ann_lrm).std() / np.sqrt(K-1)
alpha = 0.05

zL_ann_lrm = zb_ann_lrm + sig * stats.t.ppf(alpha/2, nu)
zH_ann_lrm = zb_ann_lrm + sig * stats.t.ppf(1-alpha/2, nu)
#
print([zL_ann_lrm, zH_ann_lrm])
# figure()
# boxplot(np.concatenate((ANN_Error, LRM_Error), axis=1))
# xlabel("Artificial Neural Network vs Linear Regression")
# ylabel("Generalization Error (RSM)")
# show()
#
# figure()
# boxplot(np.concatenate((ANN_Error, BL_Error), axis=1))
# xlabel("Artificial Neural Network vs Average Output")
# ylabel("Generalization Error (RSM)")
# show()
#
# figure()
# boxplot(np.concatenate((LRM_Error, BL_Error), axis=1))
# xlabel("Linear Regression vs Average Output")
# ylabel("Generalization Error (RSM)")
# show()
