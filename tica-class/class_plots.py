from class_obs_tica import ObservableTicaObject
from class_obs_tica import load_aladip
from class_obs_tica import load_met
from class_obs_tica import load_fs
from class_obs_tica import load_ddr1
import mdtraj as md

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
from matplotlib import cm


def plot_by_traj(og_data, x_transformed, y_transformed, save=False):

    """

    Parameters:
    -----------
    og_data: list of matrices
        Used to extract the indices of the trajectories
    x_transformed: list
        The points of the data projected onto the first tIC
    y_transformed: list
        The points of the data projected onto the second tIC
    save: bool
        Boolean that toggles whether or not to save the figures to the working dir (default is False)

    Output:
    ------
    No formal return, plots each of the trajectories, can turn on the option to save the figures as well

    """

    starting_idx = 0
    ending_idx = 0
    for i in range(len(og_data)):
        ending_idx += len(og_data[i])
        #     print(starting_idx, ending_idx, ending_idx - starting_idx)
        x_coors = x_transformed[starting_idx:ending_idx]
        y_coors = y_transformed[starting_idx:ending_idx]

        label = str(i + 1) + ' of ' + str(len(og_data)) + ' trajectories projected onto first two tICs'

        ax = plt.subplot(111)
        plt.scatter(x_coors, y_coors, s=0.5, alpha=0.45, c=range(len(y_coors)), cmap=cm.viridis)
        plt.xlabel('X tIC 0')
        plt.ylabel('X tIC 1')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks([-2, 0, 2])
        plt.yticks([-3, 0, 3])
        plt.title(label)
        starting_idx += len(X[i])
        plt.colorbar()
        plt.show()
        if save:
            plt.savefig(label, dpi=300)


def plot_all(x,y, save=False):
    ax = plt.subplot(111)
    plt.scatter(x, y, c=range(len(x)), cmap=cm.viridis, s=0.3, alpha=0.5)
    plt.xlabel('observable tIC 0')
    plt.ylabel('observable tIC 1')
    plt.colorbar()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks([-2, 0, 2])
    plt.yticks([-3, 0, 3])
    plt.title('projection onto first two observable tICs')
    if save:
        plt.savefig('projection_onto_2d.png', dpi=300)
    plt.show()


def plot_all_obs_cmap(x, y, obs):
    obs = np.vstack(obs)
    for i in range(len(obs[0])):
        ax = plt.subplot(111)
        plt.scatter(x, y, c=obs[:, i], cmap=cm.viridis, s=0.3, alpha=0.5)
        plt.xlabel('observable tIC 0')
        plt.ylabel('observable tIC 1')
        plt.colorbar()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks([-2, 0, 2])
        plt.yticks([-3, 0, 3])
        plt.title('projection onto first two observable tICs')
        plt.show()



def plot_3d(x,y,z, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=range(len(x)), s=.2, cmap=cm.viridis,
               alpha=.5)

    ax.set_xlabel('observable tIC0')
    ax.set_ylabel('observable tIC1')
    ax.set_zlabel('observable tIC2')
    if save:
        ax.savefig('projection_onto_3d.png', dpi=300)
    plt.show()

def clustering(x,y):
    return

def sample(traj, clusters, n_samples):
    """
    Returns a list of lists of trajs, the first dimension is the cluster, the second dim is the traj of the sample
    """
    rtn = []
    for cluster in clusters:
        cluster_samples = []
        samples = np.random.randint(0, len(cluster)-1, n_samples)
        for i in samples:
            cluster_samples.append(traj[i])
        rtn.append(cluster_samples)
    return rtn

if __name__ == '__main__':
    X, Y = load_aladip()

    a = ObservableTicaObject()
    tics = a.fit_transform(X,Y)
    # plot_all_obs_cmap(tics[:, 0], tics[:, 1], Y)

    cov = a.get_covariance()
    print(cov)

    #
    # lag = 1
    # print('Done Loading')
    #
    # mol = ObservableTicaObject(lag=lag)
    # print("Object loaded")
    #
    # print("fitting data")
    # print('Data dimensions: ', (len(X), len(X[0]), len(X[0][0])))
    # r = math.floor(len(X[0][0])/10)
    #
    # mol.fit(X, Y)
    # X_trans = mol.transform()[:, 0:3]
    #
    # # plot_by_traj(X, X_trans[:, 0], X_trans[:, 1])
    # plot_all(X_trans[:,0],X_trans[:,1])
    # plot_3d(X_trans[:,0], X_trans[:,1], X_trans[:2])
    #
    #
    # plt.show()

    # print('hi')
    #
    # filepath = '/Users/bren/desktop/msk/sonya-traj/run0-clone0.h5'
    # trajs = md.load(filepath)
    # print(type(trajs))
    # print(trajs.xyz, len(trajs), len(trajs[0]))