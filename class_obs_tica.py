from pyemma import coordinates
from pyemma.coordinates.estimation.covariance import LaggedCovariance as LC
import mdtraj as md
import numpy as np
from sklearn import decomposition
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_discrete_lyapunov
filepath = '/Users/bren/downloads/run17clone3.h5'

class ObservableTicaObject():
    def __init__(self, lag = 1, epsilon = .0001, n_components = 3):
        self.lag = lag
        self.n_components = n_components

        self.x  = None
        self.y = None

        self.x_obj = None
        self.x_whitened = None
        self.x_0 = None
        self.x_tau = None

        self.y_obj = None
        self.y_whitened = None
        self.y_0 = None
        self.y_tau = None

        self.K_xx_tuple = None
        self.K_xx = None
        self.K_xy_tuple = None
        self.K_xx_evals = None
        self.K_xy = None
        self.K_xy_s = None  # singular values for K_xy

        self.chi_bar = None
        self.gamma_bar = None

        self.O = None

        self.u = None
        self.s = None
        self.v = None

        self.x_transformed = None

        self.epsilon = epsilon

    def printall(self):
        print('______________________________ \n self.x: \n', self.x)
        print('______________________________ \n self.y: \n', self.y)

        print('______________________________ \n self.x_obj: \n', self.x_obj)
        print('______________________________ \n self.x_whitened: \n', self.x_whitened)
        print('______________________________ \n self.x_0: \n', self.x_0)
        print('______________________________ \n self.x_tau: \n', self.x_tau)

        print('______________________________ \n self.y_obj: \n', self.y_obj)
        print('______________________________ \n self.y_whitened: \n', self.y_whitened)
        print('______________________________ \n self.y_0: \n', self.y_0)
        print('______________________________ \n self.y_tau: \n', self.y_tau)

        print('______________________________ \n self.K_xx_tuple: \n', self.K_xx_tuple)
        print('______________________________ \n self.K_xx: \n', self.K_xx)
        print('______________________________ \n self.K_xy_tuple: \n', self.K_xy_tuple)
        print('______________________________ \n self.K_xy: \n', self.K_xy)
        print('______________________________ \n self.K_xy_s: \n', self.s)

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.x_obj = decomposition.PCA(whiten = True) if self.x is not None else None
        self.y_obj = decomposition.PCA(whiten = True) if self.y is not None else None

        self.x_obj.fit(np.vstack(x))
        self.y_obj.fit(np.vstack(y))

        self.whiten()
        self.estimate_koop_xx(self.x_0, self.x_tau)
        self.estimate_koop_xy(self.x_0, self.y_tau)
        self.riccati()
        self.trunc_SVD(self.O)

    def whiten(self):
        rtn = []
        try:
            self.x_whitened = [self.x_obj.transform(row) for row in self.x]
            self.x_0 = np.vstack([x[: -self.lag] for x in self.x_whitened])
            self.x_tau = np.vstack([x[self.lag:] for x in self.x_whitened])
            rtn.append((self.x_0, self.x_tau))
        except:
            raise ValueError('Need to fit x data first!')
        try:
            self.y_whitened = [self.y_obj.transform(row) for row in self.y]
            self.y_0 = np.vstack(y[: -self.lag] for y in self.y_whitened)
            self.y_tau = np.vstack(y[self.lag:] for y in self.y_whitened)
            rtn.append((self.y_0, self.y_tau))
        except:
            raise ValueError('Need to fix y data first!')

        return rtn

    def estimate_koop_xx(self, x_0, x_tau):
        self.K_xx_tuple = np.linalg.lstsq(x_0, x_tau)
        self.K_xx = self.K_xx_tuple[0]
        self.K_xx_evals = np.linalg.eigvalsh(self.K_xx)
        return self.K_xx, self.K_xx_evals

    def estimate_koop_xy(self, x_0, y_tau):
        self.K_xy_tuple = np.linalg.lstsq(x_0, y_tau)
        self.K_xy = self.K_xy_tuple[0]
        self.K_xy_s = np.linalg.svd(self.K_xy, compute_uv = False)
        return self.K_xy_s

    def riccati(self):
        if (1-self.epsilon) <= np.linalg.eigh(self.K_xx)[0][-1] <= 1 + self.epsilon:
            self.v_x = np.linalg.eigh(self.K_xx)[1][-1]
        self.chi_bar = np.vstack(self.x_whitened).mean(0)  # are these suppose to uphold K_xx.T(chi_bar) = chi_bar
        self.gamma_bar = np.vstack(self.y_whitened).mean(0)  # and K_xy.T(chi_bar) = gamma_bar

        A = self.K_xx - np.outer(self.v_x, self.chi_bar)
        B = self.K_xy - np.outer(self.v_x, self.gamma_bar)
        Q = np.dot(B, B.T)  # does this need to be multiplied in middle by inv y_whitened?
        self.O = solve_discrete_lyapunov(A,Q)
        return self.O

    def trunc_SVD(self, O):
        self.u, self.s, self.v = np.linalg.svd(self.O)

    def transform(self):
        self.x_transformed = np.dot(self.x_0, self.u.T)
        return self.x_transformed

    def fit_transform(self, x,y):
        self.fit(x,y)
        return self.transform()

if __name__ == '__main__':

    from msmbuilder.example_datasets import AlanineDipeptide
    trajs = AlanineDipeptide().get().trajectories

    from msmbuilder.featurizer import AtomPairsFeaturizer
    pairs = []
    for i in range(22):
        for j in range(i):
            pairs.append((j,i))
    X = AtomPairsFeaturizer(pairs).fit_transform(trajs)

    from msmbuilder.featurizer import DihedralFeaturizer
    Y = DihedralFeaturizer().fit_transform(trajs)
    lag = 1

    print ('\n ___________________________________________ \n')

    AlaDip = ObservableTicaObject(lag = lag)
    AlaDip.fit_transform(X, Y)
