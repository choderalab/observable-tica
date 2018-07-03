from class_obs_tica import ObservableTicaObject
import numpy as np
from sklearn import decomposition

# HELPER FUNCTIONS


def epsilon_close(a, b, epsilon = .00001):
    return True if abs(a-b) < epsilon else False


def epsilon_close_mat(a,b, epsilon=.00001):  # easy function that just checks an epsilon closeness of two similar mats
    if np.array(a).shape == np.array(b).shape:
        n = abs(a-b)
        for element in n.flatten():
            if element > epsilon:
                return False
        return True
    return 'the shapes are not the same'


def setup():
    X = np.random.randint(2, 100, (4, 100, 40))  # these two lines creates the random vectors to test on
    Y = np.random.randint(2, 100, (4, 100, 30))

    tica_obj = ObservableTicaObject()  # This chunk sets up the Obs_tICA obj so that we van run the whiten func on it
    tica_obj.x = X
    tica_obj.y = Y
    tica_obj.x_obj = decomposition.PCA(whiten=True)
    tica_obj.y_obj = decomposition.PCA(whiten=True)
    tica_obj.x_obj.fit(np.vstack(X))
    tica_obj.y_obj.fit(np.vstack(Y))
    return tica_obj


# TEST FUNCTIONS


def test_fit_kxx_shapes():  # the fit() func tests will be done after I guarantee that all the parts of fit() work first
    tica_obj = ObservableTicaObject()
    X = np.random.randint(2, 100, (4, 100, 40))
    Y = np.random.randint(2, 100, (4, 100, 30))
    tica_obj.fit(X, Y)
    shape = np.array(tica_obj.K_xx).shape
    return shape.all() == shape[0]


def test_fit_svd_shapes():
    return


def test_whiten_correctness_mean():
    X = np.random.randint(2, 100, (4, 100, 40))  # these two lines creates the random vectors to test on
    Y = np.random.randint(2, 100, (4, 100, 30))

    tica_obj = ObservableTicaObject()  # This chunk sets up the Obs_tICA obj so that we van run the whiten func on it
    tica_obj.x = X
    tica_obj.y = Y
    tica_obj.x_obj = decomposition.PCA(whiten=True)
    tica_obj.y_obj = decomposition.PCA(whiten=True)
    tica_obj.x_obj.fit(np.vstack(X))
    tica_obj.y_obj.fit(np.vstack(Y))

    tica_obj.whiten()  # Running the function

    x = np.array(np.vstack(tica_obj.x_whitened))  # Flattened versions of the whitened data
    y = np.array(np.vstack(tica_obj.y_whitened))  # for ease of cov mat calc

    x_avg = x.mean(0)  # finding the averages for x and y, post whitening
    y_avg = y.mean(0)

    assert epsilon_close_mat(x_avg, np.zeros(x_avg.shape)), "Dimensional mean of whitened X data is not zero vector"
    assert epsilon_close_mat(y_avg, np.zeros(y_avg.shape)), "Dimensional mean of whitened Y data is not zero vector"

    return "Whitened Data Mean Values Test Passed"


def test_whiten_correctness_cov():
    X = np.random.randint(2,100, (4,100,40))  # these two lines creates the random vectors to test on
    Y = np.random.randint(2,100, (4,100,30))

    tica_obj = ObservableTicaObject()  # This chunk sets up the Obs_tICA obj so that we van run the whiten func on it
    tica_obj.x = X
    tica_obj.y = Y
    tica_obj.x_obj = decomposition.PCA(whiten=True)
    tica_obj.y_obj = decomposition.PCA(whiten=True)
    tica_obj.x_obj.fit(np.vstack(X))
    tica_obj.y_obj.fit(np.vstack(Y))

    tica_obj.whiten()  # Running the function

    x = np.array(np.vstack(tica_obj.x_whitened))  # Flattened versions of the whitened data
    y = np.array(np.vstack(tica_obj.y_whitened))  # for ease of cov mat calc

    n_x = 1  # counting the number of data points to normalize the cov mats
    n_y = 1
    for i in range(len(X.shape)-1):
        n_x *= X.shape[i]
        n_y *= X.shape[i]

    cov_x = x.T.dot(x)/(n_x-1)  # '-1' for bessel correction
    cov_y = y.T.dot(y)/(n_y-1)

    assert epsilon_close_mat(cov_x, np.identity(X.shape[-1])), "Cov Mat for whitened X data is NOT identity matrix"
    assert epsilon_close_mat(cov_y, np.identity(Y.shape[-1])), "Cov Mat for whitened Y data is NOT identity"

    return 'Whitened Data Covariance Test Passed'


def test_Kxx_vals():
    X = np.random.randint(2, 100, (4, 100, 4))  # these two lines creates the random vectors to test on
    Y = np.random.randint(2, 100, (4, 100, 30))

    tica_obj = ObservableTicaObject()  # This chunk sets up the Obs_tICA obj so that we van run the whiten func on it

    tica_obj.x = X
    tica_obj.y = Y
    tica_obj.x_obj = decomposition.PCA(whiten=True)
    tica_obj.y_obj = decomposition.PCA(whiten=True)
    tica_obj.x_obj.fit(np.vstack(X))
    tica_obj.y_obj.fit(np.vstack(Y))

    tica_obj.whiten()  # Running the function
    K_xx, K_evals = tica_obj.estimate_koop_xx(tica_obj.x_0, tica_obj.x_tau)

    M = np.amax(K_evals)
    m = np.amin(K_evals)

    print('min eigenval: ', m, '\nmax eigenval: ', M)

    # The following line may not be semantically right

    assert -1 < m < M <= 1, 'Eigenvalues do not fall in the correct interval, try increasing the p fraction'

    c_0 =tica_obj.x_0.T.dot(tica_obj.x_0)
    c_tau = tica_obj.x_0.T.dot(tica_obj.x_tau)

    koop = np.linalg.inv(c_0).dot(c_tau)
    print (epsilon_close_mat(koop, K_xx))
    print (np.amin(koop), np.amax(koop))
    print ('sum.T', koop.T.sum(0))
    print('sum', koop.sum(0))
    print('det, c_0', np.linalg.det(c_0))
    print (np.linalg.eigvalsh(c_0))
    print('det, c_tau', np.linalg.det(c_tau))
    print ('det', np.linalg.det(koop))
    assert K_xx.shape == (X.shape[-1], X.shape[-1]), "K_xx is not square or does not have the right dimensions"
    assert epsilon_close(K_evals[-1], 1), "No eigenvalue with value 1"

    return "K_xx Shape and Eigenvalue Test Passed"


def test_Kxx_squared_error():
    tica_obj = setup()
    tica_obj.whiten()
    tica_obj.K_xx_tuple = tica_obj.estimate_koop_xx(tica_obj.x_0, tica_obj.x_tau)
    error_bound = np.sum((tica_obj.x_tau - tica_obj.x_0)**2)
    print ('Error upper bound: ', error_bound, '\nResidual Error: ', tica_obj.K_xx_tuple[1][0])
    assert error_bound > tica_obj.K_xx_tuple[1][0], "Residual error is greater than naive error!"

    return "Koopman Residual Error Test Passed"


def test_riccati_average_vals():
    tica_obj = setup()

    tica_obj.whiten()
    tica_obj.estimate_koop_xx(tica_obj.x_0, tica_obj.x_tau)
    tica_obj.estimate_koop_xy(tica_obj.x_0, tica_obj.y_tau)

    tica_obj.riccati()
    assert epsilon_close_mat(tica_obj.chi_bar, np.zeros(tica_obj.chi_bar.shape)), "Chi_bar is not zero vector."
    assert epsilon_close_mat(tica_obj.gamma_bar, np.zeros(tica_obj.gamma_bar.shape)), "Gamma_bar is not the zero vector"

    return "Chi_bar/Gamma_bar Test Passed"


def test_riccati_equation():
    tica_obj = setup()

    tica_obj.whiten()
    tica_obj.estimate_koop_xx(tica_obj.x_0, tica_obj.x_tau)
    tica_obj.estimate_koop_xy(tica_obj.x_0, tica_obj.y_tau)

    O = tica_obj.riccati()

    assert epsilon_close_mat(O-tica_obj.A.dot(O).dot(tica_obj.A.T), tica_obj.B.dot(tica_obj.B.T)), "Riccati was not solved correctly"

    return "Riccati Correctness Test Passed"


def test_trunc_svd():
    tica_obj = setup()

    tica_obj.whiten()
    tica_obj.estimate_koop_xx(tica_obj.x_0, tica_obj.x_tau)
    tica_obj.estimate_koop_xy(tica_obj.x_0, tica_obj.y_tau)

    O = tica_obj.riccati()
    tica_obj.trunc_SVD(O)

    assert epsilon_close_mat(tica_obj.u, tica_obj.v.T), "The matrices U and V are not transposes"
    return 'SVD test passed'

print(test_whiten_correctness_cov(), test_whiten_correctness_mean())
