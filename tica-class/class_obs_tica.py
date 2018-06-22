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

        self.v_x = None
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

        print('______________________________ \n self.chi_bar', self.chi_bar)
        print('______________________________ \n self.gamma_bar', self.gamma_bar)

        print('______________________________ \n self.O', self.O)

        print('______________________________ \n self.u', self.u)
        print('______________________________ \n self.s', self.s)
        print('______________________________ \n self.v', self.v)

        print('______________________________ \n self.x_transformed', self.x_transformed)

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

        # CHECK SHAPE OF KXX KXY AND O
        # TRUNCATED SHAPE OF V,S,U

    def whiten(self):
        rtn = []
        try:
            self.x_whitened = [self.x_obj.transform(row) for row in self.x] # change name of list comprehension unit thin
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

        # ASSERT MEANLESS AND COV = I
        # RUN THE TEST ON SELF.X_WHITENED AND NOT X_0 X_T, SAME FOR Y_WHITENED

        return rtn

    def estimate_koop_xx(self, x_0, x_tau):
        self.K_xx_tuple = np.linalg.lstsq(x_0, x_tau)
        self.K_xx = self.K_xx_tuple[0]
        self.K_xx_evals = np.linalg.eigvalsh(self.K_xx)
        return self.K_xx, self.K_xx_evals

        # CHECK  IF EIGENVALS OF K_XX FALL WITHIN INTERVALS
        # CHI_BAR, GAMMA_BAR = 0
        # CHECK SQUARED ERROR IS NO BIGGER THAN SQUARED ERROR OF X_T-X_0, ANALOG FOR K_XY

    def estimate_koop_xy(self, x_0, y_tau):
        self.K_xy_tuple = np.linalg.lstsq(x_0, y_tau)
        self.K_xy = self.K_xy_tuple[0]
        self.K_xy_s = np.linalg.svd(self.K_xy, compute_uv = False)
        return self.K_xy, self.K_xy_s

    def riccati(self):
        # print (np.linalg.eigh(self.K_xx))
        if (1-self.epsilon) <= np.linalg.eigh(self.K_xx)[0][-1] <= 1 + self.epsilon:
            self.v_x = np.linalg.eigh(self.K_xx)[1][-1]
        # else:
        #     raise  # Something bad
        self.chi_bar = np.vstack(self.x_whitened).mean(0)  # are these suppose to uphold K_xx.T(chi_bar) = chi_bar
        self.gamma_bar = np.vstack(self.y_whitened).mean(0)  # and K_xy.T(chi_bar) = gamma_bar

        # print (-self.epsilon <= self.chi_bar.any() <= self.epsilon)
        #
        # print (np.array(self.x).shape)
        # print (self.x)

        # print (self.K_xx,'\n _______________________\n', self.v_x,'\n _______________________\n', self.chi_bar)
        # print (np.linalg.eigvalsh(self.K_xx))

        A = self.K_xx - np.outer(self.v_x, self.chi_bar)
        B = self.K_xy - np.outer(self.v_x, self.gamma_bar)
        Q = np.dot(B, B.T)
        self.O = solve_discrete_lyapunov(A,Q)
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

if __name__ == '__main__':

    def load_AlaDip():

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

    def load_FS():

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

    X,Y = load_AlaDip()

    # import mdtraj as md
    #
    # refpdb = md.load('frame0.pdb')
    # trajectory = md.formats.NetCDFTrajectoryFile(netcdf_filename).read_as_traj(refpdb.topology)

    # print (type(trajectory))

    lag = 1

    print ('\n ___________________________________________ \n')

    print (len(X), len(X[0]), len(X[0][0]))
    # X = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]], [[1, 2], [3, 4], [5, 6], [7, 8]]])

    # X = np.random.randint(1,4,[3,8,3])
    #
    # Y = np.random.randint(1,4,[3,8,3])
    # print (X.shape, Y.shape)
    mol = ObservableTicaObject(lag = lag)
    mol.fit(X, Y)
    t = mol.transform()
    print (type(mol.K_xx))
    # print (np.array(AlaDip.K_xx).shape)
    print (np.linalg.norm(mol.chi_bar))
    print (np.mean(mol.chi_bar))
    # print (AlaDip.K_xx.dot(AlaDip.x_0.T).T-AlaDip.x_tau)
    # print (np.array(AlaDip.x_0).shape)
    # print (len(AlaDip.x_whitened), len(AlaDip.x_whitened[0]), len(AlaDip.x_whitened[0][0]))
    # AlaDip.printall()
    # print (len(X),len(X[0]), len(X[0][0]))
    # print (np.array(X).shape)
