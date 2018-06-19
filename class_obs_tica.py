from pyemma import coordinates
from pyemma.coordinates.estimation.covariance import LaggedCovariance as LC
import mdtraj as md
import numpy as np
from sklearn import decomposition
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_discrete_lyapunov
filepath = '/Users/bren/downloads/run17clone3.h5'

class ObservableTicaObject():
	def __init__(self, lag = 1, epsilon = .0001):
		self.lag = lag
		
		self.x  = 'hi'
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
		self.K_xy = None
		self.s = None #singular values for K_xy

		self.chi_bar = None
		self.gamma_bar = None

		self.O = None

		self.epsilon = epsilon
		self.tics = None

	def printall(self):
		print ('self.x: \n', self.x)
		print ('self.y: \n', self.y)

		print ('self.x_obj: \n', self.x_obj)
		print ('self.x_whitened: \n', self.x_whitened)
		print ('self.x_0: \n', self.x_0)
		print ('self.x_tau: \n', self.x_tau)

		print ('self.y_obj: \n', self.y_obj)
		print ('self.y_whitened: \n', self.y_whitened)
		print ('self.y_0: \n', self.y_0)
		print ('self.y_tau: \n', self.y_tau)

		# self.K_xx_tuple = None
		# self.K_xx = None
		# self.K_xy_tuple = None
		# self.K_xy = None
		# self.s = None

	def fit(self, x, y):
		self.x = x
		self.y = y
		self.x_obj = decomposition.PCA(whiten = True) if self.x is not None else None
		self.y_obj = decomposition.PCA(whiten = True) if self.y is not None else None
		try:
			self.x_obj.fit(np.vstack(x))
		except:
			print ('No x data to fit')
		try:
			self.y_obj.fit(np.vstack(y))
		except:
			print ('No y data to fit')

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

	def estimate_koop_xx(self):
		if self.x_0 is not None and self.x_tau is not None:
			self.K_xx_tuple = np.linalg.lstsq(self.x_0, self.x_tau)
			self.K_xx = self.K_xx_tuple[0]
			self.K_xx_evals = np.linalg.eigvalsh(self.K_xx)
			return (self.K_xx, self.K_xx_evals)
		else:
			print ('Need to fit and transform data first')

	def estimate_koop_xy(self):
		if self.x_0 is not None and self.y_tau is not None:
			self.K_xy_tuple = np.linalg.lstsq(self.x_0, self.y_tau)
			self.K_xy = self.K_xy_tuple[0]
			self.s = np.linalg.svd(self.K_xy, compute_uv = False)
			return self.s
		else:
			print ('Need to fit and transform data first')

	def riccati(self):
		if (1-self.epsilon) <= np.linalg.eigh(self.K_xx)[0][-1] <= 1 + self.epsilon:
			self.v_x = np.linalg.eigh(self.K_xx)[1][-1]
		self.chi_bar = np.vstack(self.x_whitened).mean(0)
		self.gamma_bar = np.vstack(self.y_whitened).mean(0)
		# np.linalg.norm(chi_bar), np.linalg.norm(gamma_bar)
		A = self.K_xx - np.outer(self.v_x, self.chi_bar)
		B = self.K_xy - np.outer(self.v_x, self.gamma_bar)
		Q = np.dot(B, B.T)
		self.O = solve_discrete_lyapunov(A,Q)
		return self.O

	def trunc_SVD(self):
		if self.O is not None:
			u, s, v = np.linalg.svd(self.O)
		else:
			print ('Have not solved riccati eq yet')

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

# print (np.array(X).shape)
AlaDip = ObservableTicaObject(lag = lag)
# print (AlaDip)
AlaDip.fit(X, Y)
# print (AlaDip.printall())
# print ('boy: \n', np.array(AlaDip.x).shape, 'girl', AlaDip.x_obj)
# AlaDip.x_whitened = AlaDip.x_obj.transform(AlaDip.x)
# print (AlaDip.x_whitened)
# print ('oy')
AlaDip.whiten()
# print (AlaDip.printall())
# print (AlaDip.printall())
AlaDip.estimate_koop_xx()
AlaDip.estimate_koop_xy()
AlaDip.riccati()
AlaDip.trunc_SVD
# print ('\n K_xx eigenvaleues: \n', AlaDip.K_xx_evals)

