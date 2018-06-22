from pyemma import coordinates
from pyemma.coordinates.estimation.covariance import LaggedCovariance as LC
import mdtraj as md
import numpy as np
from sklearn import decomposition
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_discrete_lyapunov
filepath = '/Users/bren/downloads/run17clone3.h5'

# traj_mat = md.load(filepath)

x = np.random.random((5,5))
y = np.random.random((5,4))
t = 1


# data is the data we want to fit it to, t is the time lag we desire
def observable_tica(x,y, t):
	'''
	Description:
		Performs observable tICA reduction

	Parameters: 
		x, y: matrices that you wish to perform featurization on
		t: timelag
	Returns:
		matrix of stuff
	'''
	c_xx, c_yy = whitening(x, y)
	O = riccati(c_yy, x, y, t)
	W, sigma = trunc_svd(c_xx, O)
	return W.T.dot(scipy.linalg.fractional_matrix_power(c_xx, -.5)).dot(chi(x))

# TODO: select what transformations you wish to perform on the data
def chi(x):
	return x
	feat = coordinates.featurizer(x)
	return feat.something()
def gamma(y):
	return y
	feat = coordinates.featurizer(y)
	return feat.somethingelse()


# Not sure if this whitening is doing the correct thing when I return the covariance matrix.
def whitening(x, y):
	'''
	Description:
		Spits out reduced matrices where the data is meanless and all covariance is reducted to 1.
	Parameters:
		x,y: matrices that you wish to perform featurization on through chi(x) and gamma(y) functions
	Returns:
		Covariance matrices for x and y where the autocorrelation has been reduced to zero,
		Used as an argument in the Riccati equation as well as 
	'''
	pca_obj_xx = decomposition.PCA(whiten = True)
	pca_obj_yy = decomposition.PCA(whiten = True)

	c_xx = pca_obj_xx.fit_transform(chi(x))
	c_yy = pca_obj_yy.fit_transform(gamma(y))

	return (c_xx, c_yy)

# The following function is incorrect, not sure how to return the koopman estimator, 
# also I might be able to get it out of pyemma.coordinates.tica, also unclear if these are discrete estimates
# TODO: figure out how to estimate this matrix
def koopman_est(x, y, t):
	'''
	Description:
		Function that calculates the Koopman estimators that satisfy the functions:
			chi(x_{t+1}) = (K_xx.T)(chi(x_t))
			chi(y_{t+1}) = (K_xy.T)(chi(y_t))

	Parameters:
		x,y: the matrices we want estimate the operator for
		t: timelag
	Returns:
		Two Koopman matrices that solve the above equation
	'''

	chi_data, gamma_data = whitening(x, y)

	chi_data = chi(x)
	n = len(x)

	chi_0 = chi_data[:n-t]
	chi_t = chi_data[t:]

	gamma_t = gamma(y)[t:]

	# K_xx = np.linalg.lstsq(chi_0, chi_t)[0]
	K_xy = np.linalg.lstsq(chi_0, gamma_t)[0]

	obj_x = LC(c0t = True, lag = t)

	obj_x.fit(chi(x))
	C_00, C_01 = (obj_x.C00_, obj_x.C0t_)
	K_xx = np.linalg.inv(C_00).dot(C_01)
	# K_xy = 0

	return (K_xx, K_xy)

def riccati(c_yy, x, y,t): 
	'''
	Description:
		Solves the Riccati equation, specifically AOA^T - O + Q = 0
	Parameters:
		c_yy: a whitened covariance matrix, used in calculation of Q
		x,y: matrices that are used to calculate koopman estimators
		t: timelaga
	Returns:
		The matrix, O, solved in the above equation.

	'''
	# c_xx, c_yy = whitening(x,y)
	K_xx, K_xy = koopman_est(x, y ,t)
	evals_xx,evect_xx = np.linalg.eigh(K_xx) #hermetian
	# evals_xy,evect_xy = np.linalg.eigh(K_xy)

	print ('\n', evals_xx, '\n \n', evect_xx)

	v_x = evect_xx[:,list(evals_xx).index(1)]
	try:
		v_y = np.linalg.solve(K_xy, v_x) # I think this v_y is extraneous, there isn't v_y used anywhere
	except:
		raise Error('cannot find eigenvalue = 1')
	chi_bar = np.mean(chi(x).T, axis =1)
	gamma_bar = np.mean(gamma(y).T, axis =1)


	A = K_xx - np.outer(v_x*chi_bar)
	B = K_xy - np.outer(v_x*gamma_bar)
	Q = B.dot(np.inverse(c_yy)).dot(B.T)

	return solve_discrete_lyapunov(A, Q)

def trunc_svd(c_xx, O, n_components):
	'''
	Description: 
		Performs Truncated SVD on a matrix thats computed in the function:
			C_xx^.5OC_xx^.5
	Parameters:
		c_xx: instantaneous covariance matrix for x
		O: solution to solve_discrete_lyapunov
		n_componenets: number of components
	Returns:
		svd components and singular values of the calculated matrix

	'''
	C = scipy.linalg.fractional_matrix_power(c_xx, .5)
	data = C.dot(O).dot(C)
	svd = decomposition.TruncatedSVD(n_components = n_components)
	svd.fit(data)
	return (svd.components_, svd.singular_values_)

observable_tica(x,y,t)


