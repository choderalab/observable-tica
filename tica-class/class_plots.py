from class_obs_tica import ObservableTicaObject
from class_obs_tica import load_aladip
from class_obs_tica import load_met
from class_obs_tica import load_fs

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np

from matplotlib import cm


print('Loading..')
X, Y = load_aladip()

lag = 1
print('Done Loading')

mol = ObservableTicaObject(lag=lag)
print("Object loaded")

print("fitting data")
print('Data dimensions: ', (len(X), len(X[0]), len(X[0][0])))
r = math.floor(len(X[0][0])/10)

mol.fit(X, Y)

X_trans = mol.transform()[:, 0:3]


# print(np.amin(X_trans[:,0]), np.amax(X_trans[:, 0]))
# print(np.amin(X_trans[:,1]), np.amax(X_trans[:, 1]))
# print(np.amin(X_trans[:,2]), np.amax(X_trans[:, 2]))


ax = plt.subplot(111)
plt.scatter(X_trans[:,0], X_trans[:,1], c=range(len(X_trans[:, 0])), cmap=cm.rainbow, s=0.3, alpha=0.5)
plt.xlabel('observable tIC 0')
plt.ylabel('observable tIC 1')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([-2,0,2])
plt.yticks([-3,0,3])
plt.title('projection onto first two observable tICs')
# plt.savefig('met_projection_onto_2d.png', dpi=300)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_trans[:,0], X_trans[:,1], X_trans[:, 2], c=range(len(X_trans[:,0])), s=.2, cmap=cm.tab10, alpha=1)

ax.set_xlabel('observable tIC0')
ax.set_ylabel('observable tIC1')
ax.set_zlabel('observable tIC2')

plt.show()


# print('opening files')
# f = open('tica_coords.txt', 'w+')
#
# for i in range(len(X_trans)):
#     f.write(str(X_trans[i]))
#     if i % 500 == 0:
#         print('line ', i, ' of ', len(mol.x_0))
#
# print('Done writing X_Trans')
#
# f.close()
#
# print('done')