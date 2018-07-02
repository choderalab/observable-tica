import numpy as np
from sklearn import decomposition
from scipy.linalg import solve_discrete_lyapunov
import math
filepath = '/Users/bren/downloads/run17clone3.h5'


class ObservableTicaObject:
    def __init__(self, lag=1, epsilon=.0001, n_components=3, var=.95):
        self.lag = lag
        self.n_components = n_components
        self.var = var

        self.rand_selection = None

        self.x = None
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

        self.v_x = None
        self.chi_bar = None
        self.gamma_bar = None

        self.A = None
        self.B = None
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

        print('______________________________ \n self.chi_bar', self.chi_bar)
        print('______________________________ \n self.gamma_bar', self.gamma_bar)

        print('______________________________ \n self.O', self.O)

        print('______________________________ \n self.u', self.u)
        print('______________________________ \n self.s', self.s)
        print('______________________________ \n self.v', self.v)

        print('______________________________ \n self.x_transformed', self.x_transformed)

    def fit(self, x, y, rand_selection=False):
        self.rand_selection = rand_selection

        self.x = x
        self.y = y

        self.x_obj = decomposition.PCA(whiten=True, n_components=self.var)
        self.y_obj = decomposition.PCA(whiten=True, n_components=self.var)

        print((len(x), len(x[0]), len(x[0][0])), (len(y), len(y[0]), len(y[0][0])))

        # This is for implementation of the limited sampling

        # if type(rand_selection) == int:
        #     idx = np.random.choice(len(x[0]), (1, rand_selection), replace=False)
        #
        #     x_rand = []
        #     for num in idx:
        #         x_rand.append(x[:, num].tolist())
        #     x = np.array(x_rand[0])
        #     self.x = x

        self.x_obj.fit(np.vstack(self.x))
        self.y_obj.fit(np.vstack(self.y))

        print('Whitening Data')
        self.whiten()

        print('Estimating Koopman Matrices')

        self.x_0 = np.insert(self.x_0, len(self.x_0[0]), 1, axis=1)
        self.x_tau = np.insert(self.x_tau, len(self.x_tau[0]), 1, axis=1)
        self.y_tau = np.insert(self.y_tau, len(self.y_tau[0]), 1, axis=1)

        self.estimate_koop_xx(self.x_0, self.x_tau)
        self.estimate_koop_xy(self.x_0, self.y_tau)

        print('Solving Riccati')
        self.riccati()

        print('Performing SVD')
        self.trunc_SVD(self.O)

        # CHECK SHAPE OF KXX KXY AND O
        # TRUNCATED SHAPE OF V,S,U

    def whiten(self):
        rtn = []

        try:
            self.x_whitened = [self.x_obj.transform(mat) for mat in self.x]
            self.x_0 = np.vstack([x[: -self.lag] for x in self.x_whitened])
            self.x_tau = np.vstack([x[self.lag:] for x in self.x_whitened])
            rtn.append((self.x_0, self.x_tau))
        except:
            raise ValueError('Need to fit x data first!')
        try:
            self.y_whitened = [self.y_obj.transform(traj) for traj in self.y]
            self.y_0 = np.vstack(y[: -self.lag] for y in self.y_whitened)
            self.y_tau = np.vstack(y[self.lag:] for y in self.y_whitened)
            rtn.append((self.y_0, self.y_tau))
        except:
            raise ValueError('Need to fix y data first!')

        # ASSERT MEANLESS AND COV = I
        # RUN THE TEST ON SELF.X_WHITENED AND NOT X_0 X_T, SAME FOR Y_WHITENED

        return rtn

    def estimate_koop_xx(self, x_0, x_tau):
        self.K_xx_tuple = np.linalg.lstsq(x_0, x_tau)
        self.K_xx = self.K_xx_tuple[0]
        self.K_xx_evals = np.linalg.eigvalsh(self.K_xx)
        print('largest eigenval: ', self.K_xx_evals[-1])
        return self.K_xx, self.K_xx_evals

        # CHECK  IF EIGENVALS OF K_XX FALL WITHIN INTERVALS
        # CHI_BAR, GAMMA_BAR = 0
        # CHECK SQUARED ERROR IS NO BIGGER THAN SQUARED ERROR OF X_T-X_0, ANALOG FOR K_XY

    def estimate_koop_xy(self, x_0, y_tau):
        self.K_xy_tuple = np.linalg.lstsq(x_0, y_tau)
        self.K_xy = self.K_xy_tuple[0]
        self.K_xy_s = np.linalg.svd(self.K_xy, compute_uv=False)
        return self.K_xy, self.K_xy_s

    def riccati(self):
        # if (1-self.epsilon) <= np.linalg.eigh(self.K_xx)[0][-1] <= 1 + self.epsilon:
        #     self.v_x = np.linalg.eigh(self.K_xx)[1][-1]

        self.v_x = np.linalg.eigh(self.K_xx)[1][-1]  # just setting the value of v_x to the largest evals evect
        # else:
        #     raise  # Something bad
        self.chi_bar = np.vstack(self.x_whitened).mean(0)  # are these suppose to uphold K_xx.T(chi_bar) = chi_bar
        self.gamma_bar = np.vstack(self.y_whitened).mean(0)  # and K_xy.T(chi_bar) = gamma_bar

        self.chi_bar = np.insert(self.chi_bar, len(self.chi_bar), 1)  # Adding the final element to chi and gamma bar to match shape with Kxx and Kxy
        self.gamma_bar = np.insert(self.gamma_bar, len(self.gamma_bar), 1)

        self.A = self.K_xx - np.outer(self.v_x, self.chi_bar)
        self.B = self.K_xy - np.outer(self.v_x, self.gamma_bar)
        Q = np.dot(self.B, self.B.T)
        self.O = solve_discrete_lyapunov(self.A,Q)
        return self.O
        # CHECK TO SEE IF RICCATI IS BEING SOLVED

    def trunc_SVD(self, O):
        self.u, self.s, self.v = np.linalg.svd(O)
        # TRUNCATE, TEST FOR U,S,V SHAPES, TEST FOR U == V.T

    def transform(self):
        self.x_transformed = np.dot(self.x_0, self.u)
        return self.x_transformed
        # CHECK SHAPE
        # CHECK COMPARISON BETWEEN THIS AND TICA ON X AND TICA ON Y

    def fit_transform(self, x,y):
        self.fit(x,y)
        return self.transform()


def load_aladip():

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
    return (X,Y)


def load_fs():

    from msmbuilder.example_datasets import MinimalFsPeptide
    trajs = MinimalFsPeptide().get().trajectories

    from msmbuilder.featurizer import AtomPairsFeaturizer
    pairs = []
    for i in range(264):
        for j in range(i):
            pairs.append((j, i))
    X = AtomPairsFeaturizer(pairs).fit_transform(trajs)

    from msmbuilder.featurizer import DihedralFeaturizer
    Y = DihedralFeaturizer().fit_transform(trajs)
    return (X, Y)


def load_met():
    from msmbuilder.example_datasets import MetEnkephalin
    trajs = MetEnkephalin().get().trajectories

    from msmbuilder.featurizer import AtomPairsFeaturizer
    pairs = []
    for i in range(75):
        for j in range(i):
            pairs.append((j,i))
    X = AtomPairsFeaturizer(pairs).fit_transform(trajs)

    from msmbuilder.featurizer import DihedralFeaturizer
    Y = DihedralFeaturizer().fit_transform(trajs)
    return (X,Y)


if __name__ == '__main__':

    X, Y = load_aladip()

    lag = 1

    print ('\n ___________________________________________ \n')

    print (len(X), len(X[0]), len(X[0][0]))

    mol = ObservableTicaObject(lag=lag)
    mol.fit(X, Y)
    t = mol.transform()

    print('\n ___________________________________________ \n')
    print('\n \n')